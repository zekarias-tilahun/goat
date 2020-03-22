#!/usr/bin/env bash

#python ./src/main.py --input ./data/cora/graph.txt --output-dir ./data/cora/outputs/ --tr-rate .15 --dropout 0.5 --epochs 100

python ./src/evaluate.py --te-path ./data/cora/outputs/test_graph_15.txt \
--emb-path ./data/cora/outputs/goat_context_15.emb --context True
#python ./src/main.py --input ./data/email/graph.txt --output-dir ./data/email/outputs/ --tr-rate .15 --dropout 0.8 --epochs 50

#python ./src/evaluate.py --te-path ./data/email/outputs/test_graph_15.txt --com-path ./data/email/communities.txt\
#--emb-path ./data/cora/outputs/goat_context_15.emb --context True