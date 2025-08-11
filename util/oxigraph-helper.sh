#!/bin/bash
DATASET_DIR=/local/data-ssd/sparql-benchmark/oxigraph/dblp
BENCHMARK_FILE=/local/data-ssd/sparql-benchmark/benchmark-generator/dblp.benchmark.tsv
OXIGRAPH_PATH=/local/data-ssd/sparql-benchmark/oxigraph/code/oxigraph/target/release/oxigraph
RESULT_FILE=dblp.oxigraph
MAX_MEM=32
TIMEOUT=186s  # Timeout + 5s wait after spawn + 1s buffer

start_oxigraph () {
  timeout $TIMEOUT \
    systemd-run --scope -p MemoryMax=${MAX_MEM}G -p MemoryHigh=$((MAX_MEM - 1))G --user \
    $OXIGRAPH_PATH serve-read-only -b localhost:7878 -l $DATASET_DIR &
  echo "Spawned oxigraph under systemd-run"
  sleep 5s
}

stop_oxigraph () {
  echo "Kill oxigraph"
  pkill -f $OXIGRAPH_PATH
  sleep 1s
}

run_bench () {
  # $1 = queryid
  qlever example-queries \
    --get-queries-cmd "cat $BENCHMARK_FILE" \
    --result-file $RESULT_FILE-$1 \
    --sparql-endpoint http://localhost:7878/query \
    --query-ids $1
}

NUM_QUERIES=$(cat "$BENCHMARK_FILE" | wc -l)
for i in $(seq 1 $NUM_QUERIES); do
  echo "Query $i / $NUM_QUERIES"
  start_oxigraph
  run_bench $i
  stop_oxigraph
done

# TODO: join dblp.oxigraph-*.results.yaml files into one
