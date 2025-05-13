# Sparqloscope: A generic benchmark for comprehensive performance evaluation of SPARQL engines

We provide a new benchmark, called Sparqloscope for evaluating the query performance of SPARQL engines. The benchmark has three unique features that separate it from other such benchmarks:

1. Sparqloscope is comprehensive in that it considers most features of the SPARQL 1.1 query language that are relevant in practice. In particular: basic graph patterns, OPTIONAL, FILTER, ORDER BY, LIMIT, DISTINCT, GROUP BY and aggregates, UNION, EXISTS, MINUS, SPARQL functions (for numerical values, strings, and dates).
2. Sparqloscope is generic in the sense that it can be applied to any given RDF dataset and will then produce a comprehensive benchmark for that particular dataset. Existing benchmarks are either synthetic or manually constructed for a fixed dataset.
3. Sparqloscope is specific in the sense that it aims to evaluate features in isolation (independent from other features) as much as possible. This allows pinpointing specific strengths and weaknesses of a particular engine.

Sparqloscope is free and open-source software and easy to use. As a showcase we use it to evaluate the performance of three high-performing SPARQL engines (Virtuoso, MillenniumDB, QLever) on two widely used RDF datasets (DBLP and Wikidata).

## Usage

### Example for DBLP

Assuming a SPARQL endpoint for the DBLP dataset is running on port 7015 on your machine, you can generate a benchmark for this dataset using the following command-line (for details see `--help`).
 
```bash
python3 generate-benchmark.py --sparql-endpoint http://localhost:7015 --prefix-definitions "$(cat prefixes/dblp.ttl)" --kg-name dblp
```

### Further information

Detailed setup instructions can be found in the [setup documentation](docs/setup.md).

Details on the precomputation queries, placeholders and template queries are described in [query-templates.yaml](query-templates.yaml). The exact procedure of benchmark generation is documented in [generate-benchmark.py](generate-benchmark.py).

## License

This project is licensed under the Apache 2.0 License. For more information, see the [LICENSE](LICENSE) file.
