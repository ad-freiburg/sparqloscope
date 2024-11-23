#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright 2024, University of Freiburg,
# Chair of Algorithms and Data Structures
# Author: Hannah Bast <bast@cs.uni-freiburg.de>

from __future__ import annotations

import argparse
import argcomplete
import logging
from termcolor import colored
import yaml
from SPARQLWrapper import SPARQLWrapper, JSON


# LOGGING
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


def compute_placeholders(
    placeholder_queries: list[dict[str, str]], args: argparse.Namespace
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
        description = query["description"]
        sparql_query = query["query"]
        log.info(f"Computing placeholder \"{name}\" ...")
        log.debug(sparql_query)
        sparql_endpoint.setQuery(sparql_query)
        sparql_endpoint.setReturnFormat(JSON)
        try:
            result_json = sparql_endpoint.query().convert()
            log.debug(result_json)
            first_var = result_json["head"]["vars"][0]
            binding = result_json["results"]["bindings"][0][first_var]
            if binding["type"] == "uri":
                value = f"<{binding['value']}>"
            else:
                value = binding["value"]
        except Exception as e:
            log.error(f"Error computing placeholder \"{name}\": {e}")
            exit(1)
        log.info(colored(f"{name} = {value}", "blue"))
        result[name] = value
    return result



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
        default="https://qlever.cs.uni-freiburg.de/api/dblp",
        help="The SPARQL endpoint to send the queries to",
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
    args = command_line_args()
    log.setLevel(log_levels[args.log_level])
    log.info(f"SPARQL endpoint: {args.sparql_endpoint}")
    log.info(
        f"Reading query templates and placeholder queries from "
        f"\"{args.query_templates}\" ..."
    )
    with open(args.query_templates, "r") as yaml_file:
       query_templates_yaml = yaml.safe_load(yaml_file) 
    placeholder_queries = query_templates_yaml["placeholders"]
    result = compute_placeholders(placeholder_queries, args)
    log.info(result)
