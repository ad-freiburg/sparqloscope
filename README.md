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
Wikidata, TODO: add more.
