from pathlib import Path
import argparse
import yaml


class MultiLineDumper(yaml.SafeDumper):
    def represent_scalar(self, tag, value, style=None):
        value = value.replace("\r\n", "\n")
        if isinstance(value, str) and "\n" in value:
            style = "|"
        return super().represent_scalar(tag, value, style)


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
    data["name"] = dataset
    data["description"] = data.get("index_description", "---")
    if "index_description" in data:
        del data["index_description"]
    queries = data["queries"]
    for query in queries:
        name = query["query"]
        desc = query["long_query"]
        query_ = query["sparql"]
        query["query"] = query_
        query["name"] = name
        query["description"] = desc
        del query["long_query"]
        del query["sparql"]
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
        app with updated keys. 
        """)
    arg_parser.add_argument(
        "yaml_files", nargs='+', type=str,
        help="The result yaml files to be modified.")
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = command_line_args()
    for f in args.yaml_files:
        migrate_yaml(Path(f))
