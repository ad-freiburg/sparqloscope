# SPARQL Benchmark

This is a benchmark for measuring the SPARQL query performance of a triplestore
aka SPARQL engine. It has three unique features that set it apart from other
benchmarks:

1. It does not only benchmark the performance of basic graph patterns, but
   considers all features of the SPARQL 1.1 query language that are relevant
   for performance, such as subqueries, property paths, aggregates,
   expressions, etc.

2. Each feature is tested in isolation (to provide clear evidence of the
   performance of a triplestore for that feature), as well as in combination
   with other features (wherever the combination poses challenges that are
   not simply the sum of the challenges of the individual features).

3. The benchmarks is generic and can create a concrete benchmark for any given 
   RDF dataset. For example, to generate the SPARQL query that benchmarks the
   performance of GROUP BY on the object of a predicate with many triples,
   the benchmark generator will first query the dataset to find such a
   predicate. The generation is configurable.

Concrete benchmarks are provided for the following RDF datasets: DBLP,
DBLP+Citations, Wikidata, TODO: add more.

# Quickstart

Here is the command line to generate a benchmark for the Wikidata dataset.
There are more options available, see the help message of the script or use
the autocomplete feature of your shell to see them (you might have to call
`eval "$(register-python-argcomplete generate-benchmark.py)"` first).

```
NAME=wikidata
ENDPOINT_URL=https://qlever.cs.uni-freiburg.de/api/wikidata
PREFIXES_URL=https://qlever.cs.uni-freiburg.de/api/prefixes/wikidata
python3 generate-benchmark.py --name $NAME --sparql-endpoint $ENDPOINT_URL --prefix-definitions "$(curl -s $PREFIXES_URL)"
```

This will generate an output file named $NAME.queries.tsv i.e. wikidata.queries.tsv
To have the output file in .yaml or .yml format and with a custom file name:

```
FILE_NAME=benchmark-queries.$NAME.yaml
python3 generate-benchmark.py --name $NAME --sparql-endpoint $ENDPOINT_URL --prefix-definitions "$(curl -s $PREFIXES_URL) \
--output-format yaml --output-file $FILE_NAME"
```
