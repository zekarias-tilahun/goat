# GOAT

A PyTorch implementation of the "**Go**ssip and **At**tend: Context-sensitive Graph Representation Learning" paper, to appear on the International AAAI Conference on Web and Social Media (ICWSM 2020)

**Update** The repository has been updated to support a number of other datasets and improve the quality and readability of the source code. We closely follow the best practices recommended from torchvision, PyTorch Geomtric, and OGB for the newly added dataset support. 

The code released at the time of publication can be found under the ```legacy``` directory.

Requirements!
-------------

  - Python 3.7+
  - PyTorch 1.6
  - PyTorch Geometric 1.6.1 (For datasets)
  - networkx 2.3+
  - Numpy 1.17.2+
  - Pandas 0.24.2+ 
### Requirements to run evaluation script (optional)
  - Scikit-learn 0.20.3+ 
  
  
Usage
-----

### Example usage

#### Training

```sh
$ cd goat
$ python main.py
```

Please refer to [```Training Input Arguments```](#Training-Input-Arguments) for more options

#### Evaluation (Link Prediction)

```sh
$ cd goat
$ python evaluate.py --task link_prediction
```

#### Evaluation (Node clustering)

```sh
$ cd goat
$ python evaluate.py --task node_clustering
```

#### Evaluation (Attention weight visualization)

```sh
$ cd goat
$ python evaluate.py --task visualization
```
Please refer to [```Evaluation Input Arguments```](#Evaluation-Input-Arguments) for more options

or 

The helper shell script ```goat.sh``` can be used to run all the above commands at once 

```sh
$ bash ./goat.sh
```


Training Input Arguments
------------------------


`--root:` or `-r:` 
A path to a root directory to put all the datasets. Default is ```./data```

`--name:` or `-n:`
The name of the datasets. Default is ```cora```. Check the [```Supported dataset names```](#Supported-dataset-names) 

`--dim:` or `-d:`
The size (dimension) of embedding (representation) vectors. Default is 200.

`--num-nbrs:` or `-nbr:`
The number of neighbors to be sampled. Default is 100.

`--num-neg:` or `-neg:`
The number of negative samples. Default is 2.

`--lr:` or `-lr:`
Learning rate, a value in [0, 1]. Default is 0.0001

`--dropout:` or `-do:`
Dropout rate, a value in [0, 1]. Deafult is 0.5

`--epoch:` or `-e:`
The number of epochs. Default is 10.

`--train-prop:` or `-tp:`
The proportion of training set, i.e. the fraction of edges to be used as a training set. A value in (0, 1]. Default is .50. The remaining fraction of edges (```1 - train_prop```), test edges, will be used for testing.


Evaluation Input Arguments
------------------------


`--root:` or `-r:` 
A path to a root directory to put all the datasets. Default is ```./data```

`--name:` or `-n:`
The name of the datasets. Default is ```cora```. Check the [```Supported dataset names```](#Supported-dataset-names) 

`--task:` or `-t:`
The type of evaluation task. Valid options are : ```link_prediction```, ```node_clustering```, or ```visualization```

`--epoch:` or `-e:`
The specific epoch to be evaluated. Default is 10.

`--train-prop:` or `-tp:`
The specific training proportion to be evaluated.


Supported dataset names
-----------------------
 - ```cora``` (Citation dataset - results reported in the paper)
 - ```zhihu``` (Question and answering dataset - results reported in the paper)
 - ```email``` (Email exchange dataset - results reported in the paper)
 - ```citeseer``` (Citation dataset - results reported in the paper)
 - ```pubmed``` (Citation dataset - results reported in the paper)
 - ```dblp``` (Citation dataset)
 - ```cora-full``` (Citation dataset)
 - ```computers``` (Co-purchased products from Amazon computers category)
 - ```photo``` (Co-purchased products from Amazon computers category)
 - ```physics``` (Co-Author relations)
 - ```cs``` (Co-Author relations)
 - ```flickr``` (Friendship relations)

Citing
------

If you find GOAT useful in your research, we kindly ask that you cite the following paper:

```
@misc{kefato2020gossip,
    title={Gossip and Attend: Context-Sensitive Graph Representation Learning},
    author={Zekarias T. Kefato and Sarunas Girdzijauskas},
    year={2020},
    eprint={2004.00413},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


