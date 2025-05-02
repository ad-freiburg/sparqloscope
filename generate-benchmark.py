#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright 2024 - 2025, University of Freiburg,
# Chair of Algorithms and Data Structures
# Authors: Hannah Bast <bast@cs.uni-freiburg.de>
#          Christoph Ullinger <ullingec@cs.uni-freiburg.de>

from __future__ import annotations

import argparse
import argcomplete
import logging
import sys
from contextlib import contextmanager
from typing import NotRequired, TextIO, TypedDict, Literal, Generator
from termcolor import colored
import yaml
from SPARQLWrapper import SPARQLWrapper, JSON
from pathlib import Path
import json
import re
import http.server
import socketserver
import threading


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

# Type annotations for parsed SPARQL JSONs
Binding = TypedDict("Binding", {
    "datatype": NotRequired[str],
    "type": Literal["uri"] | Literal["literal"],
    "value": str
})
Head = TypedDict("Head", {"vars": list[str]})
Results = TypedDict("Results", {"bindings": list[Binding]})
ResultJson = TypedDict("ResultJson", {"head": Head, "results": Results})


@contextmanager
def open_file_for_writing(filename: str) -> Generator[TextIO, None, None]:
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


def compute_sparql(name: str, sparql_query: str, args: argparse.Namespace) \
        -> ResultJson:
    """
    Helper function to execute a SPARQL query on the given endpoint.
    """
    log.debug(f'Computing placeholder query "{name}" ...')
    log.debug(sparql_query)
    sparql_endpoint = SPARQLWrapper(args.sparql_endpoint)
    sparql_endpoint.setQuery(sparql_query)
    sparql_endpoint.setReturnFormat(JSON)
    try:
        return sparql_endpoint.query().convert()  # type: ignore
    except Exception as e:
        log.error(f'Error executing query "{name}": {e}')
        raise


PrecomputedQuery = TypedDict("PrecomputedQuery", {
    "name": str,
    "query": str,
    "cache_result": NotRequired[bool]
})
PrecomputedQueries = list[PrecomputedQuery]
PrecomputedQueriesResult = dict[str, ResultJson]


def precompute_queries(precomputed_queries: PrecomputedQueries,
                       args: argparse.Namespace) -> PrecomputedQueriesResult:
    """
    Compute or restore the results of the queries stated in the
    precomputed_queries section of the template YAML and store the results of
    queries with cache_results flag to disk.
    """
    result = {}
    for query in precomputed_queries:
        query_name = query["name"]
        sparql_query = query["query"]
        cache = bool(query.get("cache_result"))

        # Get query result, either from cache or by requesting the endpoint
        filename = f"precomputed.{args.kg_name}.{query_name}.json"
        result_json = None
        if cache and not args.overwrite_cached_results and \
                Path(filename).exists():
            log.debug(f"Loading precomputed query result from file {filename}")
            with open(filename, "r") as f:
                result_json = json.load(f)
        else:
            log.info(f"Computing query result for {query_name}")
            result_json = compute_sparql(query_name, sparql_query, args)
            if cache:
                log.debug(
                    f"Writing precomputed query result to file {filename}")
                with open(filename, "w") as f:
                    json.dump(result_json, f)
        result[query_name] = result_json
    return result


def make_precomputed_queries_handler_class(
        precomputed_queries_result: PrecomputedQueriesResult) -> type:
    """
    Make an HTTP request handler class that can provide the precomputed SPARQL
    results as JSON. The class definition captures the
    precomputed_queries_result argument.
    """
    class PrecomputedQueriesHandler(http.server.BaseHTTPRequestHandler):
        "Custom request handler class to return SPARQL JSON results from a dict"

        def __respond(self):
            path = self.path[1:]
            if path in precomputed_queries_result:
                self.send_response(200)
                self.send_header(
                    "Content-Type", "application/sparql-results+json")
                self.end_headers()
                self.wfile.write(json.dumps(
                    precomputed_queries_result[path]).encode("utf-8"))
            else:
                self.send_response(500)
                self.end_headers()

        def do_GET(self):
            self.__respond()

        def do_POST(self):
            self.__respond()
    return PrecomputedQueriesHandler


Placeholder = TypedDict("Placeholder", {
    "name": str,
    "description": NotRequired[str],
    "query": str
})


def compute_placeholders(
    placeholders: list[Placeholder],
    precomputed_queries_result: PrecomputedQueriesResult,
    prefix_definitions: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, str]:
    """
    For each query in the `placeholders` section of the templates YAML file,
    send the query to the given SPARQL endpoint. From the result, construct the
    placeholders and return them as a dictionary.
    """

    result = {}

    def add_interal_services(query: str) -> str:
        """
        Replace placeholders for precomputed queries with the appropriate
        SPARQL SERVICE statement to retrieve the result of the precomputed
        query.
        """
        out = query
        for match in re.finditer(r'%[\w\-]+%', query):
            substr = match.group(0)
            precomputed_query = substr[1:-1]
            if precomputed_query in precomputed_queries_result:
                variables = precomputed_queries_result[
                    precomputed_query]["head"]["vars"]
                vars_str = ' '.join(f'?{v}' for v in variables)
                out = out.replace(
                    substr,
                    f"{{ SERVICE <{args.external_url}/{precomputed_query}> {{ " +
                    f"VALUES ({vars_str}) {{}} }} }}")
            elif precomputed_query in result:
                out = out.replace(substr, result[precomputed_query])
            else:
                raise AssertionError(f"Replacement {substr} unknown")
        return out

    def add_iri_brackets(binding: dict[str, str]) -> str:
        "Make a SPARQL literal from a result JSON binding"
        if binding["type"] == "uri":
            return f"<{binding['value']}>"
        else:
            return binding["value"]

    def get_placeholder_value(result_vars: list[str],
                              result_bindings: list[Binding]) \
            -> tuple[str, Binding]:
        "Extract the value for a placeholder from the result bindings"
        column = result_vars[0]
        assert len(result_bindings), "No matching binding found for placeholder"
        binding = add_iri_brackets(result_bindings[0][column])
        return (binding, result_bindings[0])

    for query in placeholders:
        p_name = query["name"]
        sparql_query = add_interal_services(query["query"])
        result_json = compute_sparql(p_name, sparql_query, args)

        result_vars = []
        result_bindings = []
        row = None
        value = None
        try:
            # For ASK queries, we want the value of the field `boolean`. For
            # SELECT queries, we want the first value of the first variable.
            if "boolean" in result_json:
                value = str(result_json["boolean"]).lower()
            else:
                result_vars = result_json["head"]["vars"]
                result_bindings = result_json["results"]["bindings"]
                log.debug(f"result_vars: {result_vars}")
                value, row = get_placeholder_value(
                    result_vars, result_bindings)
        except Exception as e:
            log.error(f'Error computing placeholder "{p_name}": {e}')
            raise

        # Log the computed value. If the result had a second variable, log that
        # as well (it is typically a count that is useful to know).
        additional_info = ""
        log.debug(f"result_bindings for used row: {row}")
        if row:
            for var, cell in row.items():
                val: Binding = cell  # type: ignore
                if "datatype" in val and val["datatype"].endswith(
                        ("#int", "#integer")):
                    additional_info += f" [{var} = {int(val['value']):,}]"

        value_disp, _ = apply_prefix_definitions(value, prefix_definitions)
        log.info(colored(f"{p_name} = {value_disp}{additional_info}", "blue"))

        # Add to result dict.
        result[p_name] = value
    return result


def apply_prefix_definitions(query: str, prefix_definitions: dict[str, str]) \
        -> tuple[str, set[str]]:
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
    output_filename: str,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    """
    Replace the placeholders in the given queries and write the queries to
    a file, in the format specified by `args.output_format`.
    """

    # Add value for special placeholder `%LIMIT%`.
    placeholders["LIMIT"] = str(args.limit)

    result = []
    num_queries_written = 0
    num_queries_error = 0
    num_queries_condition_false = 0
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
        try:
            query = re.sub(r"%([A-Z0-9_]+)%", replace_placeholder, query)
        except ValueError as e:
            log.error(e)
            num_queries_error += 1
            continue

        # Add prefix definitions and turn into a single line.
        query, prefixes_used = apply_prefix_definitions(
            query, prefix_definitions)
        query_single_line = re.sub(r"\s+", " ", query).strip()
        log.info(colored(f"{name} -> {query_single_line}", "blue"))
        if len(prefixes_used) > 0:
            prefixes_used_defs = [
                f"PREFIX {prefix}: <{prefix_definitions[prefix]}>"
                for prefix in prefixes_used
            ]
            query = f"{'\n'.join(prefixes_used_defs)}\n{query}"
            query_single_line = \
                f"{' '.join(prefixes_used_defs)} {query_single_line}"

        # For TSV, we have one description and one query per line.
        query = query.rstrip()
        query_single_line = query_single_line.rstrip()
        log.debug(colored(query, "yellow"))
        log.debug(colored(query_single_line, "yellow"))
        result.append((f"{name} [{description}]", query, query_single_line))
        num_queries_written += 1

    # Custom dumper for YAML, that dumps all values of key `query` using `|-`.
    class MultiLineDumper(yaml.Dumper):
        def represent_scalar(self, tag, value, style=None):
            if isinstance(value, str) and "\n" in value:
                style = "|"
            return super().represent_scalar(tag, value, style)

    # Write the result as TSV or YAML.
    with open_file_for_writing(output_filename) as output_file:
        if args.output_format == "tsv":
            for description, _, query in result:
                print(f"{description}\t{query}", file=output_file)
        else:
            yaml_dict = {
                "kb": args.kg_name,
                "queries": [
                    {"query": description, "sparql": query}
                    for description, query, _ in result
                ],
            }
            print(
                yaml.dump(yaml_dict, sort_keys=False, Dumper=MultiLineDumper),
                file=output_file,
            )

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
        "--kg-name",
        type=str,
        required=True,
        help="The name of the knowledge graph; the name of the "
        "output file will be based on this",
    )
    arg_parser.add_argument(
        "--output-format",
        type=str,
        choices=["tsv", "yml"],
        default="tsv",
        help="The output format for the benchmark results (tsv or yml"
        ", default: tsv)",
    )
    arg_parser.add_argument(
        "--prefix-definitions",
        type=str,
        help="Command to get prefix definitions (one per line)",
    )
    arg_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit for those queries with a LIMIT clause"
             " (default: 10)",
    )
    arg_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=log_levels.keys(),
        help=f"The log level {list(log_levels.keys())}",
    )
    arg_parser.add_argument(
        "--overwrite-cached-results",
        action="store_true",
        help="If set, all placeholder queries will be re-evaluated."
    )
    arg_parser.add_argument(
        "--external-url",
        type=str,
        default="http://localhost:8000",
        help="The URL where the SPARQL endpoint can reach this program"
    )
    arg_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port where this program can serve a SERVICE used by the " +
        "SPARQL endpoint"
    )
    arg_parser.add_argument(
        "--pause-on-service",
        action="store_true",
        help="Pause for debugging when SERVICE is started"
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
        log.info("Parsed prefix definitions " +
                 f"(#prefixes = {len(prefix_definitions)})")
        for prefix, iri in prefix_definitions.items():
            log.debug(colored(f"PREFIX {prefix}: <{iri}>", "blue"))

    # Read the YAML file.
    with open(args.query_templates, "r") as yaml_file:
        query_templates_yaml = yaml.safe_load(yaml_file)
    precomputed_queries = query_templates_yaml["precomputed_queries"]
    placeholders = query_templates_yaml["placeholders"]
    query_templates = query_templates_yaml["queries"]
    log.info(
        f"Read query templates and placeholder queries from "
        f"`{args.query_templates}` "
        f"(#precomputed queries = {len(precomputed_queries)}, "
        f"#placeholders = {len(placeholders)}, "
        f"#queries = {len(query_templates)})"
    )

    log.info("Precomputing queries for placeholder generation ...")
    precomputed_queries_result = precompute_queries(precomputed_queries, args)

    log.info("Starting HTTP server for precomputed queries ...")
    handler = make_precomputed_queries_handler_class(
        precomputed_queries_result)
    httpd = socketserver.TCPServer(("", args.port), handler)
    try:
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        log.info(f"Internal HTTP server started on port {args.port}")
        if args.pause_on_service:
            input("Paused. Press enter to continue")

        log.info("Computing placeholders ...")
        placeholders = compute_placeholders(
            placeholders, precomputed_queries_result, prefix_definitions, args)

        log.info("Generating queries ...")
        output_filename = f"{args.kg_name}.benchmark.{args.output_format}"
        num_queries_written, num_queries_error, num_queries_condition_false = (
            generate_queries(
                placeholders, query_templates, prefix_definitions,
                output_filename, args
            )
        )

        log.info(
            f"Queries written to `{output_filename}` "
            f" (format: {args.output_format}, "
            f"#written: {num_queries_written}, "
            f"#condition-false: {num_queries_condition_false}, "
            f"#errors: {num_queries_error})"
        )
    except Exception as e:
        log.error(f"Quitting because of exception: {e}")
    finally:
        httpd.shutdown()
        httpd.server_close()
