# Sparqloscope: A generic benchmark for the comprehensive and concise performance evaluation of SPARQL engines

We provide a new benchmark, called Sparqloscope, for evaluating the query performance of SPARQL engines. The benchmark combines three unique features, which separates it from other such benchmarks:

1. Sparqloscope is generic in the sense that it can be applied to any given RDF dataset and it will then produce a comprehensive benchmark for that particular dataset. Existing benchmarks are either synthetic or designed for a fixed dataset.

2. Sparqloscope is comprehensive in that it considers most features of the SPARQL 1.1 query language that are relevant in practice. In particular, it also considers advanced features like EXISTS, various SPARQL functions for numerical values, strings, and dates, language filters, etc.

3. Sparqloscope is specific in the sense that it aims to evaluate relevant features in isolation and as concisely as possible. In particular, the benchmark generated for a given knowledge graphs consists of only around 100 very carefully crafted queries, the results of which can and should be studied individually and not in aggregation.

Sparqloscope is free and open-source software and easy to use. As a showcase we use it to evaluate the performance of three high-performing SPARQL engines (Virtuoso, MillenniumDB, QLever) on two widely used RDF datasets (DBLP and Wikidata).

## Usage

### Benchmark Generation Example for DBLP

Assuming a SPARQL endpoint for the DBLP dataset is running on port 7015 on your machine, you can generate a benchmark for this dataset using the following command-line (for details see `--help`).
 
```bash
python3 generate-benchmark.py --sparql-endpoint http://localhost:7015 --prefix-definitions "$(cat prefixes/dblp.ttl)" --kg-name dblp
```

### Ready-to-use Benchmarks for Popular Datasets

You may find ready-to-use benchmarks, which we have generated using Sparqloscope for popular datasets in the [benchmarks/](benchmarks/) folder.

### Further information

An interactive web app for the evaluation results on various engines can be found at <https://purl.org/ad-freiburg/sparqloscope-evaluation>.

Detailed setup instructions for running Sparqloscope can be found in the [setup documentation](docs/setup.md).

Details on the precomputation queries, placeholders and template queries are described in [query-templates.yaml](query-templates.yaml). The exact procedure of benchmark generation is documented in [generate-benchmark.py](generate-benchmark.py).

## License

This project is licensed under the Apache 2.0 License. For more information, see the [LICENSE](LICENSE) file.
