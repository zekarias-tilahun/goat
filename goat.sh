#!/usr/bin/env bash

cd goat

python main.py -n email -tp 0.35 -do 0.8
python evaluate.py -n email -tp 0.35 -t link_prediction
python evaluate.py -n email -tp 0.35 -t node_clustering
python evaluate.py -n email -tp 0.35 -t visualization -nd 0 1