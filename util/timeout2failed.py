import yaml
import argparse


def replace_timeout_with_error(fn: str, timeout: float):
    "Load file, replace exceeded timeout with error, write to file"
    with open(fn, "r") as yaml_file:
        results = yaml.safe_load(yaml_file)
    for query in results["queries"]:
        if query["runtime_info"]["client_time"] >= timeout:
            query["headers"] = []
            query["results"] = "Timeout"
    with open(fn, "w") as yaml_file:
        print(yaml.dump(results, sort_keys=False), file=yaml_file)


def command_line_args() -> argparse.Namespace:
    """
    Parse the command line arguments and return them
    """
    arg_parser = argparse.ArgumentParser(
        description="""
        This helper takes a benchmark results yaml file and replaces all results
        exceeding the timeout with a timeout error. 
        """)
    arg_parser.add_argument(
        "yaml_file", nargs=1, type=str,
        help="The result yaml files to be modified.")
    arg_parser.add_argument(
        "--timeout", nargs=1, type=float, required=True,
        help="The benchmark timeout in seconds.")
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = command_line_args()
    replace_timeout_with_error(args.yaml_file[0], args.timeout[0])
