from sklearn import metrics, cluster

import networkx as nx
import pandas as pd
import numpy as np
import argparse

import helper


def nmi_ami(com_path, emb_path, seed=0):
    """
    Computes the normalized and adjusted mutual information scores of node embeddings in a given
    path in clustering nodes according to a ground truth community labels

    :param com_path: A path to ground truth community labels
    :param emb_path: A path to node embeddings
    :param seed: Random seed
    :return:
    """
    helper.log(f'Reading ground truth communities from {com_path}')
    com_df = pd.read_csv(com_path, header=None, sep=r'\s+', names=['node', 'com'], index_col=0)
    helper.log(f'Reading embeddings from {emb_path}')
    emb_df = pd.read_csv(emb_path, header=None, sep=r'\s+', index_col=0)

    helper.log('Building features')
    labeled_features = com_df.merge(emb_df, left_index=True, right_index=True)
    ground_truth = labeled_features.com.values
    features = labeled_features.values[:, 1:]
    num_com = len(set(ground_truth))

    helper.log(f'Learning to identify {num_com} clusters using spectral clustering')
    clustering = cluster.SpectralClustering(
        n_clusters=num_com, assign_labels="discretize", random_state=seed)
    predictions = clustering.fit(features).labels_
    nmi = metrics.normalized_mutual_info_score(ground_truth, predictions, average_method='arithmetic')
    ami = metrics.adjusted_mutual_info_score(ground_truth, predictions, average_method='arithmetic')
    return nmi, ami


def compute_link_probabilities(is_dev=True, u_embed=None, v_embed=None, test_edges=None):
    """
    Computes the link
    :param is_dev:
    :param u_embed:
    :param v_embed:
    :param test_edges:
    :return:
    """
    # Adapted from CANE: https://github.com/thunlp/CANE/blob/master/code/auc.py
    if is_dev:
        nodes = list(range(u_embed.shape[0]))
        test_edges = list(zip(range(u_embed.shape[0]), range(v_embed.shape[0])))
    else:
        nodes = list({n for edge in test_edges for n in edge})

    def get_random_index(u, v, lookup=None):
        while True:
            node = np.random.choice(nodes)
            if node != u and node != v:
                if lookup is None:
                    return node
                elif node in lookup:
                    return node

    link_probabilities = []
    for i in range(len(test_edges)):
        if is_dev:
            u = v = i
            j = get_random_index(u=i, v=i)
        else:
            u = test_edges[i][0]
            v = test_edges[i][1]
            if u not in u_embed or v not in u_embed:
                continue
            j = get_random_index(u=u, v=v, lookup=v_embed)

        u_emb = u_embed[u]
        v_emb = v_embed[v]
        j_emb = v_embed[j]

        pos_score = helper.sigmoid(u_emb.dot(v_emb.transpose()).max())
        neg_score = helper.sigmoid(u_emb.dot(j_emb.transpose()).max())

        link_probabilities.append([pos_score, neg_score])

    return np.array(link_probabilities)


def compute_lp_metrics(link_probabilities, eval_metrics=('auc', 'ap')):
    """
    Computes the link prediction scores for a given set of evaluation metrics for a specified
    link probability scores.

    :param link_probabilities: Link probability scores of true and false edges
    :param eval_metrics: A set of evaluation metrics
    :return: The dictionary containing the results for each metrics, as metric->score
    """

    results = {}
    for metric in eval_metrics:
        if metric == 'auc':
            hits = link_probabilities[:, 0] > link_probabilities[:, 1]
            hit_count = hits[hits].shape[0]

            ties = link_probabilities[:, 0] == link_probabilities[:, 1]
            tie_count = ties[ties].shape[0] / 2.
            auc = (hit_count + tie_count) / link_probabilities.shape[0]
            results['auc'] = auc
        elif metric == 'ap':
            positive_scores = zip(link_probabilities[:, 0].tolist(), [1] * link_probabilities.shape[0])
            negative_scores = zip(link_probabilities[:, 1].tolist(), [0] * link_probabilities.shape[0])
            data = sorted(list(positive_scores) + list(negative_scores), key=lambda l: l[0], reverse=True)
            data = np.array(data)
            results['ap'] = metrics.average_precision_score(data[:, 1], data[:, 0], average='micro')
    return results


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb-path', type=str, default='./data/cora/outputs/goat_context_15.emb',
                        help='Path to the embedding file')
    parser.add_argument('--te-path', type=str, default='./data/cora/outputs/test_graph_15.txt',
                        help='Path to the test edges file')
    parser.add_argument('--com-path', type=str, default='',
                        help='Path to the ground truth community file')
    parser.add_argument('--context', type=bool, default=True,
                        help='Use context-sensitive embeddings. If false aggregated embeddings will be used')
    parser.add_argument('--verbose', type=bool, default=True)
    return parser.parse_args()


def main(args):

    helper.VERBOSE = args.verbose
    if args.te_path == '' and args.com_path == '':
        helper.log('At least a path to test edges file or ground truth community file should be specified',
                   level=helper.ERROR)
    else:
        results = {}
        if args.te_path != '':
            helper.log('Running link prediction', level=helper.INFO)
            if args.context:
                embeddings = helper.read_context_embedding(args.emb_path)
            else:
                embeddings = helper.read_global_embedding(args.emb_path)
            test_graph = nx.read_edgelist(args.te_path, nodetype=int)
            auc_scores = []
            ap_scores = []
            helper.log('Computing AUC and AP scores')
            for i in range(10):
                scores = compute_link_probabilities(
                    is_dev=False, u_embed=embeddings, v_embed=embeddings,
                    test_edges=list(test_graph.edges()))
                cur_results = compute_lp_metrics(scores)
                auc_scores.append(cur_results['auc'])
                ap_scores.append(cur_results['ap'])
            avg_auc = np.mean(auc_scores)
            avg_ap = np.mean(ap_scores)

            std_auc = np.std(auc_scores)
            std_ap = np.std(ap_scores)
            helper.log(f"Mean scores out of 10 iterations:")
            helper.log(f"\tAUC score = {avg_auc}")
            helper.log(f"\tAP = {avg_ap}")
            results['link_prediction'] = avg_auc, std_auc, avg_ap, std_ap
        if args.com_path != '':
            helper.log('Running node clustering')
            nmi, ami = nmi_ami(com_path=args.com_path, emb_path=args.emb_path)
            helper.log(f'NMI: {nmi}')
            helper.log(f'AMI: {ami}')
            results['node_clustering'] = nmi, ami
        return results


if __name__ == '__main__':
    main(parse())
