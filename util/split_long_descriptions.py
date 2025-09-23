# Script to migrate result yamls to the new format expected by the web app
# (with descriptions split into short and long)
import re
from pathlib import Path
import argparse
import yaml

INDEX_DESCRIPTION = {
    "yago-4": "Full dump from https://yago-knowledge.org/downloads/yago-4, version 12.03.2020, ~2.5 billion triples",
    "dblp": "DBLP computer science bibliography, data from https://dblp.org/rdf, ~500 million triples",
    "sp2b": "SP2Bench benchmark synthetic dataset: ~50 million triples",
    "wikidata-truthy": "Wikidata Truthy from https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.gz, ~8 billion triples",
}

TIMEOUTS = {
    "yago-4": 240,
    "dblp": 180,
    "sp2b": 60,
    "wikidata-truthy": 300,
}


class MultiLineDumper(yaml.SafeDumper):
    def represent_scalar(self, tag, value, style=None):
        value = value.replace("\r\n", "\n")
        if isinstance(value, str) and "\n" in value:
            style = "|"
        return super().represent_scalar(tag, value, style)


def split_query_description(query: str) -> tuple[str, str]:
    match = re.fullmatch(r"(.+?)\s*\[(.+)\]", query)
    if match:
        short_query, long_query = match.groups()
        return short_query, long_query
    else:
        return query, ""


def migrate_yaml(yaml_file: Path):
    if yaml_file.suffix != ".yaml":
        return
    filename = yaml_file.stem
    filename_parts = filename.split(".")
    if len(filename_parts) != 3:
        return
    with open(yaml_file, "r", encoding="utf-8") as q_file:
        try:
            data = yaml.safe_load(q_file)  # Load YAML safely
        except yaml.YAMLError as exc:
            print(f"Error parsing {yaml_file} file: {exc}")
            return
    dataset = filename_parts[0]
    for index in INDEX_DESCRIPTION:
        if index in dataset:
            data["index_description"] = INDEX_DESCRIPTION[index]
            if not data.get("timeout"):
                data["timeout"] = TIMEOUTS[index]
            break
    queries = data["queries"]
    for query in queries:
        short_query_desc, long_query_desc = split_query_description(
            query["query"]
        )
        query["query"] = short_query_desc
        query["long_query"] = long_query_desc
    data["queries"] = queries
    with open(yaml_file, "w") as f:
        yaml.dump(data, f, sort_keys=False,
                  Dumper=MultiLineDumper, allow_unicode=True,)


def command_line_args() -> argparse.Namespace:
    """
    Parse the command line arguments and return them
    """
    arg_parser = argparse.ArgumentParser(
        description="""
        This helper migrates result yamls to the new format expected by the web
        app with two-part query descriptions (identifier and explanation). 
        """)
    arg_parser.add_argument(
        "yaml_files", nargs='+', type=str,
        help="The result yaml files to be modified.")
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = command_line_args()
    for f in args.yaml_files:
        migrate_yaml(f)
