# Evaluation results for ISWC'25 submission

The evaluation results for QLever, Virtuoso, MillenniumDB, GraphDB, Blazegraph and Apache Jena on DBLP and Wikidata Truthy for our paper *Sparqloscope: A generic benchmark for the comprehensive and concise performance evaluation of SPARQL engines* at the International Semantic Web Conference 2025.

## Setup

### Software

Versions:

- QLever: commit [TODO](https://github.com/ad-freiburg/qlever/tree/TODO)
- Virtuoso: version 7.2.15
- MillenniumDB: commit [ecbf6dd](https://github.com/MillenniumDB/MillenniumDB/tree/ecbf6dde5a5864f088eee3b0836ad6adba1d623b)
- GraphDB: version 11.0.0
- Blazegraph: version [2.1.6 RC](https://github.com/blazegraph/database/releases/tag/BLAZEGRAPH_2_1_6_RC)
- Apache Jena: version 5.5.0

All engines were installed natively on an Ubuntu 24.04 LTS system and configured as described in the setup documentation.

### Hardware

The machive we used is equipped with an AMD Ryzen 9 9950X CPU (16 cores, 32 threads, 5.8 GHz), 190 GiB of DDR5 memory and four 8 TB NVMe disks in a RAID 0 configuration.
