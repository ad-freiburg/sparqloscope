# Setup documentation for Sparqloscope

To setup Sparqloscope, you need to install the benchmark generator. Since our benchmark is dataset-dependent, you also need to provide a running SPARQL engine for the dataset on which you wish to run benchmarks, at the time of benchmark generation. Using the following step-by-step guide, you can setup and run our benchmark for Wikidata Truthy and DBLP on the SPARQL engines QLever, Virtuoso and MillenniumDB.   

## 1. Installation and preparation (once)

### 1.1. Download the benchmark generator

Download the software in this repository and install the required python packages:

```bash
git clone https://github.com/ad-freiburg/sparqloscope.git
cd sparqloscope
python3 -m venv .venv
source .venv/bin/activate
pip install argcomplete termcolor pyyaml
```

### 1.2. Install the qlever-control script

```bash
pip install qlever
# or if your OS package manager controls python packages:
pipx install qlever
```

For details see the [qlever-control](https://github.com/ad-freiburg/qlever-control) repository.

### 1.3. Install QLever and further SPARQL engines natively

For details on building QLever, please refer to the [QLever repository](https://github.com/ad-freiburg/qlever), especially the [Dockerfile](https://github.com/ad-freiburg/qlever/blob/master/Dockerfile) contains the commands required to build QLever on an Ubuntu system. Note that it is recommended to install QLever natively using analogous commands instead of running it through Docker to avoid distorting results with containerization overhead.

In order to setup the other SPARQL engines to be evaluated in the benchmark, please review their documentation:

- MillenniumDB: <https://github.com/MillenniumDB/MillenniumDB>
- Virtuoso: <https://github.com/openlink/virtuoso-opensource>

Like with QLever, it is recommended to install the engines natively on your system instead of using containers.

In the following steps, we assume that all three engines are properly installed and available on your PATH.

## 2. Download and index the dataset you wish to use

### 2.1. Download the dataset

Download the dataset you wish to use in Turtle or N-triples format. For the datasets we used the files can be found at:

- DBLP: <https://doi.org/10.4230/dblp.rdf.ntriples.2025-04-01>
- Wikidata Truthy: <https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.gz>

### 2.2. Build indices for the data on each engine

We recommend that you use a separate directory for each combination of an engine and a dataset. We indexed our datasets using the default settings of all engines. For details on how to index a given dataset, consult the documentation of the respective engine.

## 3. Generate the benchmark using Sparqloscope

### 3.1. Start QLever for benchmark generation

For the benchmark generation, we use QLever. Please start it using `qlever start` in the data directory.

To ensure that the benchmark generation queries work, please set these settings:

```bash
qlever settings group-by-hash-map-enabled=true
qlever settings default-query-timeout=24h
qlever settings service-max-value-rows=0
qlever settings cache-service-results=true
```

### 3.2. Run Sparqloscope

When the QLever instance for the dataset `DATASET` is running on port `PORT` and `PORT2` is currently unused and visible to the SPARQL engine, you can now execute Sparqloscope using:

```bash
python3 generate-benchmark.py \
  --sparql-endpoint http://localhost:PORT \
  --prefix-definitions "$(cat prefixes/DATASET.ttl)" \
  --kg-name DATASET \
  --external-url http://localhost:PORT2 \
  --port PORT2
```

For more readable benchmark queries, please provide the prefix declarations for your dataset under `prefixes/DATASET.ttl`, otherwise remove the option `--prefix-definitons`.

### 3.3. Stop QLever

To stop the QLever instance used for benchmark generation, simply execute `qlever stop` in QLever's data directory.

## 4. Execute the benchmark and view results

### 4.1. Configure and cold-start the respective engine

Apply the configuration parameters suitable for the given dataset. For our demonstration datasets, you can find the settings we used below. We start each engine directly before our benchmark runs, so that the engine has empty caches.

#### Recommended configuration for DBLP

- **QLever**: in the `Qleverfile`, set `SYSTEM = native`, `MEMORY_FOR_QUERIES = 26G`, `CACHE_MAX_SIZE = 6G` and `TIMEOUT = 180s` and apply `qlever settings group-by-hash-map-enabled=true` after engine start
- **Virtuoso**: in the `virtuoso.ini` file, set `NumberOfBuffers = 2720000`, `MaxDirtyBuffers = 2000000` (the recommended values for 32 GiB of RAM) and `MaxQueryExecutionTime = 180`
- **MillenniumDB**: start the server with these flags `mdb-server --timeout 180 --threads 2 --versioned-buffer 20GB --unversioned-buffer 2GB --private-buffer 2GB --strings-static 4GB --strings-dynamic 4GB .`

#### Recommended configuration for Wikidata Truthy

- **QLever**: in the `Qleverfile`, set `SYSTEM = native`, `MEMORY_FOR_QUERIES = 54G`, `CACHE_MAX_SIZE = 10G`, `CACHE_MAX_SIZE_SINGLE_ENTRY = 5G` and `TIMEOUT = 300s` and apply `qlever settings group-by-hash-map-enabled=true` after engine start
- **Virtuoso**: in the `virtuoso.ini` file, set  `NumberOfBuffers = 5450000`, `MaxDirtyBuffers = 4000000` (the recommended values for 64 GiB or RAM) and `MaxQueryExecutionTime = 300`
- **MillenniumDB**:  start the server with these flags `mdb-server --timeout 300 --threads 2 --versioned-buffer 52GB --unversioned-buffer 2GB --private-buffer 2GB --strings-static 5GB --strings-dynamic 3GB .`

### 4.2. Execute the benchmark

To run the benchmark, you can use the `qlever` script for all engines as follows, where `DATASET` is the name of your dataset, `ENGINE` is the name of the engine you wish to benchmark and `ENDPOINT` is the URL of the SPARQL endpoint provided by this engine (`http://localhost:PORT/sparql` for MillenniumDB and Virtuoso and `http://localhost:PORT` for QLever):

```bash
qlever example-queries --get-queries-cmd "cat DATASET.benchmark.tsv" --result-file DATASET.ENGINE --sparql-endpoint ENDPOINT
```

This will result in a file `DATASET.ENGINE.results.yaml` containing all the benchmarks, the running times and results for this engine.

### Repeat 4.1.-4.2. for each engine

### 4.3. View results using the evaluation web app (optional)

You may use the `qlever` script to view an interactive web app of the evaluation results using `qlever serve-evaluation-app --results-dir DIR`, where `DIR` is the directory with your `*.results.yaml` files.
