#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright 2025, University of Freiburg,
# Chair of Algorithms and Data Structures
# Author: Christoph Ullinger <ullingec@cs.uni-freiburg.de>

import argparse
import json
import argcomplete
import re
import yaml
from typing import Optional
from pathlib import Path
import csv
import statistics

RESULTS_YAML_FILENAME_RE = re.compile(
    r'^(?P<dataset>[a-zA-Z\-]+)\.(?P<engine>[a-zA-Z\-]+)\.results\.ya?ml$')
BAD_COL_NAME_RE = re.compile(r'[^a-zA-Z]+')

# Type for a parsed result: float represents seconds, None represents error
SingleDatasetAndEngineResultsDict = dict[str, Optional[float]]
# Dataset name, engine name and results dict
SingleDatasetAndEngineResults = tuple[str, str,
                                      SingleDatasetAndEngineResultsDict]


def load_results_yaml(filename: str) \
        -> SingleDatasetAndEngineResults:
    """
    Load a results yaml file and extract the required values.
    """
    fn_match = RESULTS_YAML_FILENAME_RE.match(Path(filename).name)
    assert fn_match, \
        "The filename should be of the form DATASET.ENGINE.results.yaml " + \
        "and match the regular expression: " + RESULTS_YAML_FILENAME_RE.pattern
    with open(filename, "r") as yaml_file:
        res = yaml.safe_load(yaml_file)
    dataset = fn_match.group("dataset")
    engine = fn_match.group("engine")
    query_results: SingleDatasetAndEngineResultsDict = {}
    for query_result in res["queries"]:
        name = query_result["query"]
        if query_result["headers"] == [] and \
                isinstance(query_result["results"], str):
            # Error
            qtime = None
        else:
            qtime = query_result["runtime_info"]["client_time"]
        query_results[name] = qtime
    return dataset, engine, query_results


# Dict maps query names to results for each (dataset, engine)
AllResultsDict = dict[str, dict[tuple[str, str], Optional[float]]]


def merge_multiple_results(parsed: list[SingleDatasetAndEngineResults]) \
        -> AllResultsDict:
    """
    Merge and invert SingleDatasetAndEngineResults for output
    """
    by_query: AllResultsDict = {}
    for dataset, engine, query_results in parsed:
        for query_name, query_time in query_results.items():
            if query_name not in by_query:
                by_query[query_name] = {}
            by_query[query_name][(dataset, engine)] = query_time
    return by_query


def filter_results_dict(inp: AllResultsDict, filters: dict[str, str]):
    """
    Filters the AllResultsDict in-place by applying the regex filters on
    each query name.
    """

    assert len(filters)

    # Compile regexes
    rexps: list[tuple[re.Pattern, str]] = []
    for re_str, new_name in filters.items():
        rexps.append((re.compile(re_str), new_name))

    # Apply filters
    out = {}
    for query_name in list(inp.keys()):
        for r, new_name in rexps:
            if r.search(query_name):
                if not new_name:
                    new_name = query_name
                assert new_name not in out, f"{new_name} in {repr(out)}"
                out[new_name] = inp[query_name]
                break
    return out


def make_column_name(dataset: str, engine: str) -> str:
    """
    Make a safe column name that is suitable for LaTeX csvreader (that is,
    contains only letters a-z and A-Z)
    """
    def words(x: str) -> str:
        return "".join(w.lower().capitalize() for w in x.split("-"))
    clean_d = BAD_COL_NAME_RE.sub("x", words(dataset))
    clean_e = BAD_COL_NAME_RE.sub("x", words(engine))
    return clean_d[0].lower() + clean_d[1:] + clean_e.capitalize()


def make_aggregated(inp: AllResultsDict, timeouts: dict[str, int],
                    penalties: list[float]) \
        -> dict[str, dict[tuple[str, str], str]]:
    # Key: dataset, engine -> Value: Times for each query (None=failure)
    per_ds_engine: dict[tuple[str, str], list[Optional[float]]] = {}
    # Key: dataset, query -> Value: fastest engine name, time
    fastest: dict[tuple[str, str], tuple[Optional[str], Optional[float]]] = {}

    for query_name, query_results in inp.items():
        for ds_engine, qtime in query_results.items():
            if ds_engine not in per_ds_engine:
                per_ds_engine[ds_engine] = []
            per_ds_engine[ds_engine].append(qtime)

            # Fastest
            dataset = ds_engine[0]
            k = (dataset, query_name)
            if k not in fastest:
                fastest[k] = None, None
            _, prev = fastest[k]
            if prev is None or (qtime is not None and qtime < prev):
                fastest[k] = ds_engine[1], qtime

    fastest_ds_engine = {k: 0 for k in per_ds_engine}
    total = {k[0]: 0 for k in per_ds_engine}
    for (dataset, _), (engine, _) in fastest.items():
        if engine is None:
            continue
        fastest_ds_engine[(dataset, engine)] += 1
        total[dataset] += 1

    agg_res = {
        "percentage failed": {
            k: f"{(sum(1 for i in v if i is None) / len(v))*100:.1f}\\%"
            for k, v in per_ds_engine.items()}
        # "percentage fastest": {
        #     k: f"{(fastest_ds_engine[k] / total[k[0]])*100:.2f}\\%"
        #     for k in per_ds_engine},
    }
    for penalty in penalties:
        agg_res |= {
            f"geometric mean (penalty {penalty})": {
                k: f"{statistics.geometric_mean(i if i is not None else penalty * timeouts[k[0]] for i in v):.2f}"
                for k, v in per_ds_engine.items()
            }
        }
    agg_res |= {
        "median": {
            k: f"{statistics.median(i if i is not None else penalties[0] * timeouts[k[0]] for i in v):.2f}"
            for k, v in per_ds_engine.items()
        }
    }
    return agg_res


def make_csv_head(inp: AllResultsDict, rownamecol: str = "queryname",
                  add_fastest_col: bool = True) \
        -> list[str]:
    """
    Generate the CSV head for csv.DictWriter
    """
    res = [rownamecol]
    for dataset, engine in next(iter(inp.values())):
        f = make_column_name(dataset, "fastest")
        if add_fastest_col and f not in res:
            res.append(f)
        res.append(make_column_name(dataset, engine))
    assert len(res) == len(set(res)), \
        "Multiple columns in the output would be mapped to the same " + \
        "name due to bad characters in the input filenames"
    return res


def write_csv(inp: AllResultsDict, filename: str):
    """
    Write the results to a csv file with the given name.
    """
    with open(filename, "w") as f:
        writer = csv.DictWriter(f, make_csv_head(inp), quoting=csv.QUOTE_NONE)
        writer.writeheader()
        for query_name, query_results in inp.items():
            row = {"queryname": query_name}
            dataset_min: dict[str, Optional[float]] = {}
            for (dataset, engine), qtime in query_results.items():
                if dataset not in dataset_min:
                    dataset_min[dataset] = qtime
                else:
                    prev = dataset_min[dataset]
                    if qtime is not None and (prev is None or qtime < prev):
                        dataset_min[dataset] = qtime
                val = f"{qtime:.2f}" if qtime else "Error"
                row[make_column_name(dataset, engine)] = val
            for dataset, minimum in dataset_min.items():
                row[make_column_name(dataset, "fastest")] = \
                    f"{minimum:.2f}" if minimum is not None else "-1"
            writer.writerow(row)


def write_agg_csv(inp: AllResultsDict, filename: str, timeouts: dict[str, int],
                  penalty: list[float]):
    """
    Write the aggregated results to a csv file with the given name.
    """
    with open(filename, "w") as f:
        writer = csv.DictWriter(f, make_csv_head(
            inp, "metricname", False), quoting=csv.QUOTE_NONE)
        writer.writeheader()
        for metric, row in make_aggregated(inp, timeouts, penalty).items():
            writer.writerow({make_column_name(*k): v for k, v in row.items()} | {
                            "metricname": metric})


def command_line_args() -> argparse.Namespace:
    """
    Parse the command line arguments and return them
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "yaml_files", nargs='+', type=str,
        help="""The result yaml files produced by the 'qlever' command-line
        script during execution of a benchmark. Their filenames must be of the
        form DATASET.ENGINE.results.yaml.""")
    arg_parser.add_argument(
        "--filter-queries", nargs='?', type=str,
        help="""
        Filename of JSON dictionary: Regular expressions to match query names,
        optionally mapping to renaming of queries for output table (otherwise
        to value null).
        """
    )
    arg_parser.add_argument(
        "--output", "-o", nargs=1, type=str,
        help="The filename for the output CSV file.",
        default="benchmark_result.csv"
    )
    arg_parser.add_argument(
        "--output-agg", "-oa", nargs=1, type=str,
        help="The filename for the aggregated data output CSV file.",
        default="benchmark_aggregated.csv"
    )
    arg_parser.add_argument(
        "--query-timeouts", "-qt", nargs=1, type=str,
        help="A JSON dictionary mapping datasets to query timeouts in seconds.",
        default=json.dumps({"dblp": 180, "wikidata-truthy": 300})
    )
    arg_parser.add_argument(
        "--error-penalty", "-ep", nargs='*', type=float,
        help="In aggregated time statistics, timeout * penalty will be used for"
        "failed queries.",
        default=[2, 10]
    )

    argcomplete.autocomplete(arg_parser, always_complete_options="long")
    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    """
    Main procedure
    """
    # Parse the command line arguments.
    args = command_line_args()

    # Read input
    parsed: list[SingleDatasetAndEngineResults] = []
    for fn in args.yaml_files:
        parsed.append(load_results_yaml(fn))

    # Merge input
    merged = merge_multiple_results(parsed)
    write_agg_csv(merged, args.output_agg[0],
                  json.loads(args.query_timeouts), args.error_penalty)

    # Apply filters
    if args.filter_queries:
        with open(args.filter_queries, "r") as f:
            merged = filter_results_dict(merged, json.load(f))

    # Make csv output files
    write_csv(merged, args.output[0])
