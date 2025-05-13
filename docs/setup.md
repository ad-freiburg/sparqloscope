# Setup documentation for Sparqloscope

To setup Sparqloscope, you need to install the benchmark generator. Since our benchmark is dataset-dependent, you also need to provide a running SPARQL engine for the dataset on which you wish to run benchmarks at the time of benchmark generation. Using the following step-by-step guide, you can setup and run our benchmark for Wikidata Truthy and DBLP on the SPARQL engines QLever, Virtuoso and MillenniumDB.   

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

For details on building QLever, please refer to the [QLever repository](https://github.com/ad-freiburg/qlever), especially the [Dockerfile](https://github.com/ad-freiburg/qlever/blob/master/Dockerfile) contains the commands required to build QLever on an Ubuntu system. Note that it is recommended to install QLever natively using analogous commands instead of running it through Docker to avoid distorting results using containerization overhead.

In order to setup the other SPARQL engines to be evaluated in the benchmark, please review their documentation:

- MillenniumDB: <https://github.com/MillenniumDB/MillenniumDB>
- Virtuoso: <https://github.com/openlink/virtuoso-opensource>

Like with QLever, it is recommended to install the engines natively on your system instead of using containers.

In the following steps, we assume that all three engines are properly installed and available on your PATH.

## 2. Download and index the dataset you wish to use

### 2.1. Download the dataset

DBLP, Wikidata Truthy

*TODO*

### 2.2. Build indices for the data on each engine

- Config for the engines
- 

## 3. Generate the benchmark using Sparqloscope

### 3.1. Start QLever for benchmark generation

qlever settings group-by-hash-map-enabled=true
qlever settings default-query-timeout=24h

### 3.2. Run Sparqloscope

### 3.3. Stop QLever

## 4. Execute the benchmark and view results

### 4.1. Clear operating system disk cache

### 4.2. Configure and cold-start the respective engine

#### Recommended configuration for DBLP

QLever:

MEMORY_FOR_QUERIES = 26G
CACHE_MAX_SIZE     = 6G

Virtuoso:

Settings for 32 GB of RAM:

Mdb:


### 4.3. Execute the benchmark

### Repeat 4.1.-4.3. for each engine

### 4.4. View results using the evaluation web app (optional)

