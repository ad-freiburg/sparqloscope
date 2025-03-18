#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright 2024, University of Freiburg,
# Chair of Algorithms and Data Structures
# Author: Hannah Bast <bast@cs.uni-freiburg.de>

from __future__ import annotations

import argparse
import argcomplete
import logging
import sys
from contextlib import contextmanager
from typing import TextIO
from termcolor import colored
import yaml
from SPARQLWrapper import SPARQLWrapper, JSON
import re


class LogFormatter(logging.Formatter):
    """
    Custom formatter for logging.
    """

    def format(self, record):
        message = record.getMessage()
        if record.levelno == logging.DEBUG:
            return colored(f"{message}", "magenta")
        elif record.levelno == logging.WARNING:
            return colored(f"{message}", "yellow")
        elif record.levelno in [logging.CRITICAL, logging.ERROR]:
            return colored(f"{message}", "red")
        else:
            return message


# Setup the logger.
log = logging.getLogger("sparql-benchmark")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(LogFormatter())
log.addHandler(handler)
log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "NO_LOG": logging.CRITICAL + 1,
}


@contextmanager
def open_file_for_writing(filename: str) -> TextIO:
    """
    Context manager for opening a file for writing or using STDOUT.
    """
    if filename == "-":
        yield sys.stdout
    else:
        with open(filename, "w") as file:
            yield file


def parse_prefix_definitions(prefix_definitions: str) -> dict[str, str]:
    """
    Parse the prefix definitions from the given string (one prefix definition
    per line) and return them as a dictionary.
    """
    result = {}
    for i, line in enumerate(prefix_definitions.split("\n")):
        keyword, prefix, iri = line.split()
        if keyword not in ["PREFIX", "@prefix"]:
            raise ValueError(
                f"Error parsing prefix definitions, line {i+1}: "
                f"Expected `PREFIX` or `@prefix`, got `{keyword}`"
            )
        prefix_regex = re.compile(r"^[a-zA-Z0-9_]+:$")
        if not re.match(prefix_regex, prefix):
            raise ValueError(
                f"Error parsing prefix definitions, line {i+1}: "
                f"should match `{prefix_regex.pattern}`, got `{prefix}`"
            )
        iri_regex = re.compile(r"^<[^>]+>$")
        if not re.match(iri_regex, iri):
            raise ValueError(
                f"Error parsing prefix definitions, line {i+1}: "
                f"should match `{iri_regex.pattern}`, got `{iri}`"
            )
        result[prefix[:-1]] = iri[1:-1]
    return result


def compute_placeholders(
    placeholder_queries: list[dict[str, str]],
    prefix_definitions: dict[str, str],
    args: argparse.Namespace
) -> dict[str, str]:
    """
    For each query in the `placeholders` section of the given YAML file,
    send the query to the given SPARQL endpoint. Return the results as a
    dictionary.
    """
    sparql_endpoint = SPARQLWrapper(args.sparql_endpoint)
    result = {}
    for query in placeholder_queries:
        name = query["name"]
        sparql_query = query["query"]
        log.debug(f'Computing placeholder "{name}" ...')
        log.debug(sparql_query)
        sparql_endpoint.setQuery(sparql_query)
        sparql_endpoint.setReturnFormat(JSON)
        result_vars = []
        result_bindings = []
        try:
            result_json = sparql_endpoint.query().convert()
            log.debug(result_json)
            # For ASK queries, we want the value of the field `boolean`. For
            # SELECT queries, we want the first value of the first variable.
            if "boolean" in result_json:
                value = str(result_json["boolean"]).lower()
            else:
                result_vars = result_json["head"]["vars"]
                result_bindings = result_json["results"]["bindings"]
                first_var = result_vars[0]
                binding = result_bindings[0][first_var]
                if binding["type"] == "uri":
                    value = f"<{binding['value']}>"
                    value, _ = apply_prefix_definitions(value, prefix_definitions)
                else:
                    value = binding["value"]
        except Exception as e:
            log.error(f'Error computing placeholder "{name}": {e}')
            exit(1)
        # Log the computed value. If the result had a second variable, log that
        # as well (it is typically a count that is useful to know).
        additional_info = ""
        log.debug(f"result_bindings: {result_bindings}")
        log.debug(f"result_vars: {result_vars}")
        if len(result_bindings) > 0 and len(result_vars) > 1:
            var2 = result_vars[1]
            var2_binding = result_bindings[0][var2]
            value2 = var2_binding["value"]
            if var2_binding["datatype"].endswith(("#int", "#integer")):
                value2 = f"{int(value2):,}"
            additional_info = f" [{var2} = {value2}]"
        log.info(colored(f"{name} = {value}{additional_info}", "blue"))
        # Store the result.
        result[name] = value
    return result


def apply_prefix_definitions(query: str, prefix_definitions: dict[str, str]) -> str:
    """
    Check which of the given prefix definitions can be used in the query. For
    each such prefix definition, abbreviate the corresponding IRIs. Return the
    modified query and the set of prefixes used.
    """
    prefixes_used = set()
    for prefix, iri in prefix_definitions.items():
        if f"<{iri}" in query:
            query = re.sub(r"<" + iri + "([^/>]*)>", f"{prefix}:\\g<1>", query)
            prefixes_used.add(prefix)
    return query, prefixes_used


def generate_queries(
    placeholders: dict[str, str],
    query_templates: list[dict[str, str]],
    prefix_definitions: dict[str, str],
    args: argparse.Namespace,
) -> None:
    """
    Replace the placeholders in the given queries and write the queries to
    standard output, in the format specified by `args.output_format`.
    """
    if args.output_format != "tsv":
        log.error(
            f"Unsupported output format: {args.output_format}"
            " (currently only TSV is supported)"
        )
        exit(1)
    num_queries_written = 0
    num_queries_error = 0
    num_queries_condition_false = 0
    with open_file_for_writing(args.output_file) as file:
        for query_template in query_templates:
            # Get the values for the various fields (`condition` is optional).
            name = query_template["name"]
            description = query_template["description"]
            # group = query_template["group"]
            query = query_template["query"]
            condition = query_template.get("condition", None)

            # If there is a condition, it must be one of the placeholders and
            # its value must be `true` or `false`. If that is no the case, we
            # report an error. If the condition evaluates to `false`, we skip
            # the query.
            if condition:
                if condition not in placeholders:
                    log.error(
                        f"Error processing query template `{name}`: "
                        f"Condition `{condition}` must be one of the "
                        f"placeholders but is not, skipping query"
                    )
                    num_queries_error += 1
                    continue
                if placeholders[condition] not in ["true", "false"]:
                    log.error(
                        f"Error processing query template `{name}`: "
                        f"Condition `{condition}` must evaluate to "
                        f"either `true` or `false` but is "
                        f"`{placeholders[condition]}`, skipping query"
                    )
                    num_queries_error += 1
                    continue
                if placeholders[condition] == "false":
                    log.info(
                        colored(
                            f"Skipping query `{name}` because condition "
                            f"`{condition}` evaluates to `false`",
                            "magenta",
                        )
                    )
                    num_queries_condition_false += 1
                    continue

            # Helper function for replacing a placeholder. Throws an exception
            # if the placeholder is not defined.
            def replace_placeholder(match):
                placeholder = match.group(1)
                if placeholder in placeholders:
                    return placeholders[placeholder]
                else:
                    raise ValueError(
                        f"Error processing query template `{name}`: "
                        f"Placeholder `%{placeholder}%` is not defined, "
                        f"skipping query"
                    )

            # Iterate over the placeholders in the query, check that they are
            # defined, and replace each by the respective value.
            log.debug(f"Replacing placeholders in query: {query}")
            try:
                query = re.sub(r"%([A-Z0-9_]+)%", replace_placeholder, query)
            except ValueError as e:
                log.error(e)
                num_queries_error += 1
                continue

            # Add prefix definitions and turn into a single line.
            query, prefixes_used = apply_prefix_definitions(query, prefix_definitions)
            query_single_line = re.sub(r"\s+", " ", query).strip()
            log.info(colored(f"{name} -> {query_single_line}", "blue"))
            if len(prefixes_used) > 0:
                prefixes_used_defs = " ".join(
                    [
                        f"PREFIX {prefix}: <{prefix_definitions[prefix]}>"
                        for prefix in prefixes_used
                    ]
                ).strip()
                query_single_line = f"{prefixes_used_defs} {query_single_line}"
                log.debug(colored(query_single_line, "blue"))

            # For TSV, we have one description and one query per line.
            print(f"{name} [{description}]\t{query_single_line}", file=file)
            num_queries_written += 1

    # Return statistics.
    return (num_queries_written, num_queries_error, num_queries_condition_false)


def command_line_args() -> argparse.Namespace:
    """
    Parse the command line arguments and return them.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--query-templates",
        type=str,
        default="query-templates.yaml",
        help="The YAML file with the templates for the benchmark queries "
        "and the queries to compute the placeholders",
    )
    arg_parser.add_argument(
        "--sparql-endpoint",
        type=str,
        required=True,
        help="The SPARQL endpoint for computing the placeholders",
    )
    arg_parser.add_argument(
        "--output-file",
        type=str,
        default="benchmark-queries.tsv",
        help="The output file for the generated queries"
        " (use `-` for standard output)",
    )
    arg_parser.add_argument(
        "--output-format",
        type=str,
        choices=["tsv"],
        default="tsv",
        help="The output format for the benchmark results"
        "; currently the only supported format is TSV (as required by "
        "the `qlever example-queries` command)",
    )
    arg_parser.add_argument(
        "--prefix-definitions",
        type=str,
        help="Command to get prefix definitions (one per line)",
    )
    arg_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=log_levels.keys(),
        help=f"The log level {list(log_levels.keys())}",
    )
    argcomplete.autocomplete(arg_parser, always_complete_options="long")
    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    """
    Main function.
    """
    # Parse the command line arguments.
    args = command_line_args()
    log.setLevel(log_levels[args.log_level])
    log.info(f"SPARQL endpoint: {args.sparql_endpoint}")

    # Parse the prefix definitions (if provided).
    prefix_definitions = {}
    if args.prefix_definitions:
        prefix_definitions = parse_prefix_definitions(args.prefix_definitions)
        log.info(
            f"Parsed prefix definitions " f"(#prefixes = {len(prefix_definitions)})"
        )
        for prefix, iri in prefix_definitions.items():
            log.debug(colored(f"PREFIX {prefix}: <{iri}>", "blue"))

    # Read the YAML file.
    with open(args.query_templates, "r") as yaml_file:
        query_templates_yaml = yaml.safe_load(yaml_file)
    placeholder_queries = query_templates_yaml["placeholders"]
    query_templates = query_templates_yaml["queries"]
    log.info(
        f"Read query templates and placeholder queries from "
        f"`{args.query_templates}` "
        f"(#placeholders = {len(placeholder_queries)}, "
        f"#queries = {len(query_templates)})"
    )

    log.info("Computing placeholders ...")
    placeholders = compute_placeholders(placeholder_queries,
                                        prefix_definitions, args)

    log.info("Generating queries ...")
    num_queries_written, num_queries_error, num_queries_condition_false = (
        generate_queries(placeholders, query_templates,
                         prefix_definitions, args)
    )

    log.info(
        f"Queries written to `{args.output_file}` "
        f" (format: {args.output_format}, "
        f"#written: {num_queries_written}, "
        f"#condition-false: {num_queries_condition_false}, "
        f"#errors: {num_queries_error})"
    )
