# GOAT

A PyTorch implementation of the "```Go```ssip and ```At```tend: Context-sensitive Graph Representation Learning" paper, to appear on the International AAAI Conference on Web and Social Media (ICWSM 2020)

### Requirements!
  - Python 3.6+
  - PyTorch 1.3.1+
  - networkx 2.3+
  - Numpy 1.17.2+
  - Scikit-learn 0.20.3+ (optional, used for evaluation)
  - Pandas 0.24.2+ (optional, used for evaluation)
  
### Usage
#### Example usage

```sh
$ bash ./goat.sh
```

or 

```sh
$ python src/main.py
```

### Input Arguments


`--input:`
A path to a graph file. Default is ```./data/cora/graph.txt```

`--fmt:`
The format of the input graph, either ```edgelist``` or ```adjlist```. Default is ```edgelist```

`--output-dir:`
A path to a directory to save intermediate and final outputs of GOAT. Default is ```./data/cora/outputs```

`--dim:`
The size (dimension) of nodes' embedding (representation) vector. Default is 200.

`--epochs:`
The number of epochs. Default is 100.

`--tr-rate:`
Training rate, i.e. the fraction of edges to be used as a training set. A value in (0, 1]. Default is .15. The remaining fraction of edges (```1 - tr_rate```), test edges, will be saved in the directory specified by ```--ouput-dir``` argument.

`--dev-rate:`
Development rate, i.e. the fraction of the training set to be used as a development (validation) set. A value in [0, 1). Default is 0.2.

`--learning-rate:`
Learning rate, a value in [0, 1]. Default is 0.0001

`--dropout-rate:`
Dropout rate, a value in [0, 1]. Deafult is 0.5

`--nbr-size:`
The number of neighbors to be sampled. Default is 100.

`--directed:`
Whether the graph is directed or not. 1 for directed and 0 for undirected. Default is 1.

`--workers:`
The number of parallel workers. Default is 8.

`--verbose:`. 
Whether to turn on a verbose logger or not. 1 is on and 0 is off. Default is 1.

Citing
------

If you find GOAT useful in your research, we kindly ask that you cite the following paper:

```
@inproceedings{ZekariasICWSM2020,
  author    = {Zekarias T. Kefato and
               Sarunas Girdzijauskasr},
  title     = {Gossip and Attend: Context-sensitive Graph Representation Learning},
  booktitle = {in Proceedings of the International AAAI Conference on Web and Social Media (Association for the Advancement of Artificial Intelligence, 2020).},
  year      = {2020},
  month     = June,
  type      = {CONFERENCE},
}
```


