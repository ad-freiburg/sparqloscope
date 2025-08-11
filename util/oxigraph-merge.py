import yaml
y = {
    "timeout": 180,
    "queries": []
}
for i in range(1, 106):
    with open(f"dblp.oxigraph-{i}.results.yaml", "r") as yaml_file:
        x = yaml.safe_load(yaml_file)
        y["queries"].extend(x["queries"])

print(yaml.dump(y, sort_keys=False))
