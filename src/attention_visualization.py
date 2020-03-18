import numpy as np
import argparse

import os


def min_max(X, new_max, new_min):
    mn = X.min()
    mx = X.max()
    X_std = (X - mn) / (mx - mn + 0.000000001)
    return X_std * (new_max - new_min) + new_min
    
def attention_visualization(path, neighborhood, weights, node_id, prefix="", mode='a', names=None):
    weights_ = min_max(weights[weights!=0], new_max=1, new_min=0)
    neighborhood_ = neighborhood[neighborhood!=-1].astype('int')
    
    def format_output(j):
        node = names[neighborhood_[j]] if names is not None else neighborhood_[j]
        if (j + 1) % 15 == 0:
            return f"<br><span style='background-color: rgba(255, 0, 0, {weights_[j]})'>{node}</span>"
        else:
            return f"<span style='background-color: rgba(255, 0, 0, {weights_[j]})'>{node}</span>"
        
    with open(path, mode) as f:
        nodes = set()
        output = f'<p>Attention weights of {prefix} node neighborhood</p> ' + ' '.join(format_output(i) for i in range(len(neighborhood_)))
        nid = f'<br>Gossiper as {prefix}: {node_id if names is None else names[node_id]}'
        f.write(f"{nid}{output}<br><br>")
        
        
def load(args, comment='#'):
    atn = np.load(args.input)
    if args.names is not None and args.names != "":
        node_names = {}
        with open(args.names) as f:
            for line in f:
                if not line.startswith('#'):
                    ln = line.strip().split()
                    node_names[int(ln[0])] = '_'.join(ln[1:])
        return atn, node_names
    return atn, None
        
def main():
    args = parse_args()
    atn, node_names = load(args)
    # keys
    # ['source', 'target', 'source_neighborhood', 'target_neighborhood', 'source_attentions', 'target_attentions']
    source = atn['source']
    target = atn['target']
    source_neighborhood = atn['source_neighborhood']
    target_neighborhood = atn['target_neighborhood']
    source_attentions = atn['source_attentions']
    target_attentions = atn['target_attentions']
    
    if args.s < 0:
        index = np.random.randint(0, source.shape[0])
    else:
        #1515
        src_locs = np.where(source==args.s)[0]
        if args.t < 0:
            index = np.random.choice(src_locs)
        else:
            trg_locs = np.where(target==args.t)[0]
            index = set(src_locs) & set(trg_locs)
            if len(index) == 0:
                raise ValueError(f"Unable to find connecting edge between source node {args.s} and target node {args.t}")
            else:
                index = np.random.choice(list(index))
          
    print(f'Attention visualization for edge ({source[index]}, {target[index]})')
    
    src = source[index]
    trg = target[index]
    src_nh = source_neighborhood[index]
    trg_nh = target_neighborhood[index]
    src_nh_atn = source_attentions[index]
    trg_nh_atn = target_attentions[index]
    lst = list(zip(source, target))
    print(lst)
    path = os.path.join(args.output, f'attention_vis_{src}_{trg}.html')
    attention_visualization(path, neighborhood=src_nh, weights=src_nh_atn,
                            node_id=src, prefix='source', mode='w', names=node_names)
    attention_visualization(path, neighborhood=trg_nh, weights=trg_nh_atn, 
                            node_id=trg, prefix='target', mode='a', names=node_names)
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input', default='./data/cora/outputs/attention_weights.npz',
                        help='Path to a numpy npz file containing attention weights')
    parser.add_argument('--names', default='', help='A path to node names file')
    parser.add_argument('--output', default='./data/cora/outputs/',
                        help='A directory to write the attention visualization.')
    parser.add_argument('--s', default=-1, type=int, 
                        help='Id of a source node to inspect. If a negative value is specified a random source node will be selected. Default is -1.')
    parser.add_argument('--t', default=-1, type=int, 
                        help='Id of a target node to inspect. If a negative value is specified a random target node will be selected. Default is -1.')
    return parser.parse_args()
    
    
main()