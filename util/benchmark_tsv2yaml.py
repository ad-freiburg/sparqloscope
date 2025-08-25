import yaml
import argparse
import re

QUERY_NAME = re.compile(r'^(?P<name>.+) \[(?P<description>.+)\]$')


class MultiLineDumper(yaml.SafeDumper):
    "Custom dumper for YAML, that dumps all values of key `query` using `|-`."

    def represent_scalar(self, tag, value, style=None):
        value = value.replace("\r\n", "\n")
        if isinstance(value, str) and "\n" in value:
            style = "|"
        return super().represent_scalar(tag, value, style)


def command_line_args() -> argparse.Namespace:
    """
    Parse the command line arguments and return them
    """
    arg_parser = argparse.ArgumentParser(
        description="This helper converts a TSV benchmark file to YAML.")
    arg_parser.add_argument("tsv_file", type=str, help="The source TSV file.")
    arg_parser.add_argument("--kg-title", "-t", type=str, required=True)
    arg_parser.add_argument("--kg-description", "-d", type=str)
    return arg_parser.parse_args()


def convert(tsv: str, title: str, description: str) -> str:
    lines = tsv.splitlines(keepends=False)
    yaml_dict = {
        "title": title,
        "description": description,
        "queries": [],
    }
    for line in lines:
        name, query = line.split("\t")
        m = QUERY_NAME.match(name)
        assert m
        yaml_dict["queries"].append({
            "name": m.group("name"),
            "description": m.group("description"),
            "query": query,
        })
    return yaml.dump(yaml_dict, sort_keys=False, Dumper=MultiLineDumper)


if __name__ == "__main__":
    args = command_line_args()
    with open(args.tsv_file) as f:
        tsv = f.read()
    print(convert(tsv, args.kg_title, args.kg_description))
