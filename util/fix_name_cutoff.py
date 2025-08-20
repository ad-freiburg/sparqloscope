
import yaml
import sys
import re
from pathlib import Path

SPLITTER = re.compile(r"\.\.\.| \[")

ALL_BENCHMARK_QUERIES = []

# Custom dumper for YAML, that dumps all values of key `query` using `|-`.


class MultiLineDumper(yaml.SafeDumper):
    def represent_scalar(self, tag, value, style=None):
        value = value.replace("\r\n", "\n")
        if isinstance(value, str) and "\n" in value:
            style = "|"
        return super().represent_scalar(tag, value, style)


with open(Path(__file__).parent / "../query-templates.yaml", "r") as f:
    t = yaml.safe_load(f)
    for q in t["queries"]:
        ALL_BENCHMARK_QUERIES.append((q["name"], q["description"]))

for f in sys.argv[1:]:
    j = 0
    with open(f, "r") as yf:
        x = yaml.safe_load(yf)
    for i, q in enumerate(x["queries"]):
        name = q["query"]
        s = SPLITTER.split(name)
        if len(s) == 1:
            continue
        if s[-1].strip().endswith("]"):
            s[-1] = s[-1].strip()[:-1]
        print(j, name, s)
        while not ALL_BENCHMARK_QUERIES[j][0].startswith(s[0].strip()) or not ALL_BENCHMARK_QUERIES[j][1].endswith(s[-1]):
            j += 1
        q["query"] = f"{ALL_BENCHMARK_QUERIES[j][0]} [{ALL_BENCHMARK_QUERIES[j][1]}]"
        print("-->", q["query"])
        j += 1
    with open(f, "w") as yf:
        yaml.dump(x, yf, sort_keys=False,
                  Dumper=MultiLineDumper, allow_unicode=True,)
