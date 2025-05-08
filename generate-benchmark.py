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
from typing import Iterator, NotRequired, TextIO, TypedDict, Literal, \
    Generator, Optional
from termcolor import colored
import yaml
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from pathlib import Path
import json
import re
import http.server
import socketserver
import threading
import time


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

# Cache version
CACHE_VERSION_PLACEHOLDERS = 1
CACHE_VERSION_ARGMAXS = 0


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


def add_url_param(url: str, param_key: str, param_value: str) -> str:
    "Safely add a parameter to a given URL"
    parsed = urlparse(url)
    parameters = parse_qs(parsed.query)
    parameters[param_key] = [param_value]
    new_url = urlunparse(parsed._replace(
        query=urlencode(parameters, doseq=True)))
    return new_url


def compute_sparql(name: str, sparql_query: str,
                   args: argparse.Namespace,
                   timeout_sec: Optional[int] = None) -> ResultJson:
    """
    Helper function to execute a SPARQL query on the given endpoint.
    """
    log.debug(f'Computing result for "{name}" ...')
    log.debug(sparql_query)
    url = args.sparql_endpoint
    if timeout_sec is not None:
        url = add_url_param(url, "timeout", f"{timeout_sec}s")
    req = Request(url, data=sparql_query.encode(), headers={
        "Accept": "application/sparql-results+json",
        "Content-type": "application/sparql-query"
    })
    try:
        s = time.monotonic()
        with urlopen(req, timeout=timeout_sec) as response:
            qr = response.read().decode('utf-8')
        log.debug('End to end time of ' +
                  f'{name}: {(time.monotonic() - s) * 1000:.0f} ms')
        return json.loads(qr)
    except HTTPError as e:
        error_body = e.read().decode('utf-8')
        log.error(
            f"HTTP Error {e.code} ({e.reason}) during execution of SPARQL " +
            f"query {name}:\n {error_body}")
        raise
    except Exception as e:
        log.error(
            f'Error executing query for "{name}" on SPARQL endpoint '
            + f'{args.sparql_endpoint}: {e}')
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
        cache_dir = Path("precomputed-cache")
        cache_dir.mkdir(exist_ok=True)
        if cache and not args.overwrite_cached_results and \
                (Path(filename).exists() or
                 Path(cache_dir, filename).exists()):
            if Path(cache_dir, filename).exists():
                filename = Path(cache_dir, filename)
            log.debug(f"Loading precomputed query result from file {filename}")
            with open(filename, "r") as f:
                result_json = json.load(f)
        else:
            log.info(f"Computing query result for {query_name}")
            result_json = compute_sparql(query_name, sparql_query, args)
            if cache:
                log.debug(
                    f"Writing precomputed query result to file {filename}")
                with open(Path(cache_dir, filename), "w") as f:
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


PlaceholderChild = TypedDict("PlaceholderChild", {
    "variable": str,
    "suffix": str
})
Placeholder = TypedDict("Placeholder", {
    "name": str,
    "description": NotRequired[str],
    "query": str,
    "argmax": NotRequired[str],
    "children": NotRequired[list[PlaceholderChild]],
    "multiplaceholder": NotRequired[bool]
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
                    f"{{ SERVICE <{args.external_url}/{precomputed_query}>" +
                    f" {{ VALUES ({vars_str}) {{}} }} }}")
            elif precomputed_query in result:
                out = out.replace(substr, result[precomputed_query])
            else:
                raise AssertionError(f"Replacement {substr} unknown")
        return out

    def get_precomputed_queries_mtimes(query: str) -> Iterator[int]:
        "Get modification times of precomputed queries for cache integrity"
        for match in re.finditer(r'%[\w\-]+%', query):
            substr = match.group(0)
            precomputed_query = substr[1:-1]
            if precomputed_query in precomputed_queries_result:
                fn = Path(
                    f"precomputed.{args.kg_name}.{precomputed_query}.json")
                if fn.exists():
                    yield round(fn.stat().st_mtime)
                elif Path("precomputed-cache", fn).exists():
                    yield round(Path("precomputed-cache", fn).stat().st_mtime)
                else:
                    log.debug(
                        f"Precomputed query {fn} not found. Invalidating " +
                        "placeholder cache.")
                    yield round(time.time())

    def add_iri_brackets(binding: dict[str, str]) -> str:
        "Make a SPARQL literal from a result JSON binding"
        if binding["type"] == "uri":
            return f"<{binding['value']}>"
        else:
            return binding["value"]

    def get_placeholder_values(result_vars: list[str],
                               result_bindings: list[Binding],
                               column: Optional[str] = None) \
            -> Iterator[tuple[str, int]]:
        "Extract the values for a placeholder from all result bindings rows"
        # If no variable is given, use the first one
        if column is None:
            column = result_vars[0]
        elif column.startswith("?"):
            column = column[1:]

        assert len(result_bindings), "No matching binding found for placeholder"
        assert all(column in row for row in result_bindings), \
            f"Column '{column}' not present in query result"

        # Generator for pairs of placeholder value and source row number
        return (
            (add_iri_brackets(row[column]), i)
            for i, row in enumerate(result_bindings))

    def evaluate_argmax(p_name: str, argmax: str,
                        candidates: list[dict[str, str]]) -> Optional[int]:
        """
        If a placeholder query has multiple result rows, the argmax query
        is used to assign a score to each result. The final placeholder will be
        argmax over this score.
        """
        assert candidates, "Argmax query needs at least one candidate"
        _max: Optional[float] = None
        _max_i: Optional[int] = None

        log.debug(f"Evaluating argmax for {p_name}")
        cache_dir = Path("argmax-cache")
        cache_dir.mkdir(exist_ok=True)
        cert_cache = {}
        query_cache_pth = cache_dir / \
            Path(f"argmax.{args.kg_name}.{p_name}.json")
        if not args.overwrite_cached_results and query_cache_pth.exists():
            with open(query_cache_pth, "r") as f:
                cert_cache = json.load(f)
            if cert_cache.get("_version") != CACHE_VERSION_ARGMAXS:
                log.debug(
                    f"Discarding cache {query_cache_pth}, " +
                    "because of version mismatch")
                cert_cache = {}

        error_count = 0

        for i, bindings in enumerate(candidates):
            # Replace names of placeholders with values of current candidate
            c = argmax
            for suffix, value in bindings.items():
                c = c.replace(f"%{p_name}{suffix}%", value)
            log.debug(f"Argmax query for candidate: {repr(bindings)}")

            # Evaluate argmax query
            if c in cert_cache:
                result_json = cert_cache[c]
                log.debug(
                    f"Reusing cached query result for argmax query {i}")
            else:
                try:
                    result_json = compute_sparql(
                        f"{p_name}_argmax_{i}", c, args, args.argmax_timeout)
                    cert_cache[c] = result_json
                except Exception as e:
                    log.warning(
                        f"Skipping this argmax candidate due to error: {e}")
                    error_count += 1
                    continue

            result_var_ = result_json["head"]["vars"][0]
            result_bindings_ = result_json["results"]["bindings"]

            # Check result has expected form
            if not (len(result_bindings_) == 1 and
                    result_var_ in result_bindings_[0] and
                    "datatype" in result_bindings_[0][result_var_] and
                    result_bindings_[0][result_var_]["datatype"]
                    .endswith(("#int", "#integer", "#decimal"))):
                error_count += 1
                log.warning(
                    "Argmax queries must return a single numeric score " +
                    f"value, but query for {p_name} returned " +
                    f"{repr(result_json)}")
                continue

            # Extract score from result of argmax query
            val = float(result_bindings_[0][result_var_]["value"])
            log.debug(f"Argmax {i} score: {val}")
            if _max is None or val > _max:
                _max, _max_i = val, i

        # Store computed argmaxs for this placeholder to cache file
        with open(query_cache_pth, "w") as f:
            json.dump(cert_cache | {"_version": CACHE_VERSION_ARGMAXS}, f)
            log.debug(f"Wrote cached argmax queries to {query_cache_pth}")

        # Return row index for placeholder query result with maximum argmax
        # score
        assert _max_i is not None and _max is not None, \
            f"No argmax result could be found. {error_count} queries were " + \
            "skipped due to errors."
        log.debug(f"Argmax maximum row is {_max_i} with score {_max}")
        return _max_i

    placeholder_cache_dir = Path("placeholder-cache")
    placeholder_cache_dir.mkdir(exist_ok=True)

    for query in placeholders:
        p_name = query["name"]
        precomputed_mtimes = list(
            get_precomputed_queries_mtimes(query["query"]))

        cache_pth = Path(placeholder_cache_dir, f"placeholder.{p_name}.json")
        cached = None
        if not args.overwrite_cached_results and cache_pth.exists():
            with open(cache_pth, "r") as f:
                cached = json.load(f)
            # Cache does not match
            if cached.get("query") != query:
                cached = None
                log.debug(
                    "Discarding cached placeholder data because " +
                    "configuration of cached placeholder does not match " +
                    "placeholder in template yaml file.")
            elif cached.get("_version") != CACHE_VERSION_PLACEHOLDERS:
                cached = None
                log.debug("Discarding cached placeholder becasue of version " +
                          "mismatch.")
            elif cached.get("precomputed_mtimes") != precomputed_mtimes:
                cached = None
                log.debug("Discarding cached placeholder because at least " +
                          "one of the used placeholder queries has been " +
                          "recomputed since caching.")

        row = None
        values: dict[str, str] = {}

        if cached:
            log.debug(f"Restoring cached placeholder from {cache_pth}")
            row = cached["row"]
            values = cached["values"]
        else:
            # If the placeholder has children, this means from the same row of
            # the placeholder query, multiple bindings will become placeholders.
            # This is required for example to obtain matching predicates for
            # join operations.
            children: Optional[dict[str, str]] = None
            if "children" in query:
                children = {}
                for child in query["children"]:
                    var = child["variable"]
                    if var.startswith("?"):
                        var = var[1:]
                    children[var] = child["suffix"]

            # Log current computation as this may be a longer running task
            res_pl = [
                p_name + s for s in children.values()
            ] if children else [p_name]
            log.info(colored(
                f"Computing placeholder{'' if len(res_pl) == 1 else 's'} " +
                f"{', '.join(res_pl)}...", "blue"))
            log.debug(f"Placeholder Children: {repr(children)}")

            # Get the result of the main placeholder query. This may contain
            # multiple rows and cols of interest.
            sparql_query = add_interal_services(query["query"])
            result_json = compute_sparql(p_name, sparql_query, args)

            argmax_raw = query.get("argmax")

            result_vars = []
            result_bindings = []

            try:
                # For ASK queries, we want the value of the field `boolean`. For
                # SELECT queries, we want the first value of the first variable.
                if "boolean" in result_json:
                    values = {"": str(result_json["boolean"]).lower()}
                else:
                    result_vars = result_json["head"]["vars"]
                    result_bindings = result_json["results"]["bindings"]
                    n_rows = len(result_bindings)
                    log.debug(f"result_vars: {result_vars}")

                    # Extract all possible values for each child placeholder
                    # from the query result
                    possible_values_per_child = {}
                    if not children:
                        children = {result_vars[0]: ""}
                    for column, suffix in children.items():
                        possible_values_per_child[suffix] = list(
                            get_placeholder_values(
                                result_vars, result_bindings, column))

                    assert not (argmax_raw and query.get("multiplaceholder")), \
                        "You may not use the argmax and multiplaceholder " + \
                        "options together in one placeholder configuration"
                    if argmax_raw:
                        # We use a argmax to determine which row of the
                        # placeholder query will be used to set the placeholder
                        # child values

                        # Candidates for argmax
                        candidates = []
                        for i in range(n_rows):
                            candidate = {}
                            for suffix in children.values():
                                candidate[suffix] = \
                                    possible_values_per_child[suffix][i][0]
                                candidates.append(candidate)
                        # Run argmax queries
                        row_number = evaluate_argmax(p_name,
                                                     argmax_raw,
                                                     candidates)
                        assert row_number is not None
                        row = result_bindings[row_number]
                        # Values for each of the child placeholders
                        values = {
                            suffix: add_iri_brackets(row[column])
                            for column, suffix in children.items()
                        }
                    else:
                        # If no argmax is used: must be one result row or
                        # declared as a multiplaceholder
                        if not query.get("multiplaceholder"):
                            assert n_rows == 1, \
                                "If no argmax query is used and the " + \
                                "placeholder is not declared as a " + \
                                "multiplaceholder, the result " + \
                                "of a placeholder query must have " + \
                                f"exactly one row, but query for {p_name}" + \
                                f" had {n_rows} rows."
                            row = result_bindings[0]
                            # Values for each of the child placeholders
                            values = {
                                suffix: add_iri_brackets(row[column])
                                for column, suffix in children.items()
                            }
                        else:
                            # Multiplaceholder
                            values = {
                                f"{suffix}_{i}": add_iri_brackets(row[column])
                                for column, suffix in children.items()
                                for i, row in enumerate(result_bindings)
                            } | {"_NUM": str(n_rows)}

            except Exception as e:
                log.error(f'Error computing placeholder "{p_name}": {e}')
                raise

            with open(cache_pth, "w") as f:
                json.dump({
                    "_version": CACHE_VERSION_PLACEHOLDERS,
                    "precomputed_mtimes": precomputed_mtimes,
                    "query": query,
                    "row": row,
                    "values": values
                }, f)
                log.debug(f"Wrote cached placeholder result to {cache_pth}")

        # Log the computed values. If the result had further numeric variables,
        # log that as well (it is typically a count that is useful to know).
        additional_info = ""
        log.debug(f"Placeholder values: {repr(values)}, " +
                  f"result_bindings for used row: {row}")
        if row:
            for var, cell in row.items():
                val: Binding = cell  # type: ignore
                if "datatype" in val:
                    if val["datatype"].endswith(("#int", "#integer")):
                        additional_info += f" [{var} = {int(val['value']):,}]"
                    if val["datatype"].endswith(("#numeric", "#decimal")):
                        additional_info += \
                            f" [{var} = {float(val['value']):.2f}]"

        # Save placeholder values for each child
        for suffix, value in values.items():
            value_disp, _ = apply_prefix_definitions(value, prefix_definitions)
            log.info(colored(
                f"{p_name}{suffix} = {value_disp}{additional_info}",
                "blue"))

            # Add to result dict.
            result[p_name + suffix] = value
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

        # Resolve loops for multiplaceholders first
        m = list(re.finditer(r'%begin-foreach:(?P<name>[A-Z0-9_]+)%', query))
        assert len(m) <= 1, \
            "Currently at most one foreach per template is allowed"
        if m:
            begin = m[0]
            end = re.search(r'%end-foreach:(?P<name>[A-Z0-9_]+)%', query)
            assert begin and end and begin.group("name") == end.group("name"), \
                "Foreach begin and end don't match"
            foreach_name = begin.group("name")

            # Replace foreach
            def replace_foreach(match: re.Match) -> str:
                num_pl = foreach_name + "_NUM"
                assert num_pl in placeholders, \
                    f"Loop range placeholder {num_pl} not defined"
                assert re.match(r'\d+$', placeholders[num_pl]), \
                    f"{num_pl} must be a positive integer"
                n = int(placeholders[num_pl])
                body = match.group("body")
                res = ""
                for i in range(n):
                    def replace_foreach_loop_var(match: re.Match) -> str:
                        if match.group("modifier"):
                            return str(i + int(match.group("modifier")))
                        return str(i)
                    res += re.sub(r'%#i(?P<modifier>[+-]?\d+)?%',
                                  replace_foreach_loop_var, body)
                return res
            query = re.sub(
                r'%begin-foreach:[A-Z0-9_]+%(?P<body>(.|\n|\r)*)' +
                r'%end-foreach:[A-Z0-9_]+%', replace_foreach, query)

        # Helper function for replacing a placeholder. Throws an exception
        # if the placeholder is not defined.
        def replace_placeholder(match: re.Match) -> str:
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
    arg_parser = argparse.ArgumentParser(
        description="""
        SPARQL Benchmark Generator: Apply generic benchmark templates to a
        concrete knowledge graph.
        """
    )
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
    arg_parser.add_argument(
        "--argmax-timeout",
        type=int,
        default=300,
        help="Discard placeholder candidates if the argmax query exceeds this "
        "amout of seconds in runtime."
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
    try:
        httpd = socketserver.TCPServer(("", args.port), handler)
    except OSError as e:
        log.error(
            f"Could not start internal HTTP server on port {args.port}: {e}")
        exit(1)
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
        httpd.socket.close()
