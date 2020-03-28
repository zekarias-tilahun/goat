from data import compile_training_data

import torch

import numpy as np

import evaluate
import helper
import model

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


def to_cpu_tensor(tensor):
    if device == 'cpu':
        return tensor
    return tensor.cpu().data.numpy()


class GoatWrapper:

    def __init__(self, args, train_graph=None, test_graph=None):
        self._args = args
        self.data = compile_training_data(args, train_graph=train_graph, test_graph=test_graph)
        self.loss_fun = model.LogLoss  # AblationLogLoss
        self.model = None
        self.context_embedding = {}
        self.global_embedding = {}
        self.attention_data = {
            'source': [], 'target': [], 'source_neighborhood': [], 'target_neighborhood': [],
            'source_attentions': [], 'target_attentions': []}

    def _validate(self, train_loss, epoch):
        args = self._args
        losses = []
        aucs = []
        for batch in self.data.dev_batches:
            _, _, source_rep, target_rep = self._infer(batch=batch)
            val_crt = self.loss_fun(self.model)
            link_probs = evaluate.compute_link_probabilities(u_embed=source_rep, v_embed=target_rep)
            cur_results = evaluate.compute_lp_metrics(link_probs, {'auc'})

            losses.append(to_cpu_tensor(val_crt.loss))
            aucs.append(cur_results['auc'])

        helper.log('Epoch: {}/{} training loss: {:.5f} validation loss: {:.5f} validation AUC: {:.5f}'.format(
            epoch + 1, args.epochs, train_loss, np.mean(losses), np.mean(aucs)))

    def train(self):
        args = self._args
        self.model = model.GOAT(num_nodes=self.data.num_nodes, emb_dim=args.dim)
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        total_batches_ = self.data.total_batches
        bunch = total_batches_ / 100
        intervals = np.arange(0, total_batches_, bunch)

        helper.log(f'Total batches per epoch {total_batches_}')

        for i in range(args.epochs):
            for batch in self.data.train_batches:
                self._infer(batch)
                criterion = self.loss_fun(self.model)
                optimizer.zero_grad()
                criterion.loss.backward()
                optimizer.step()

            if args.dev_rate > 0:

                self._validate(train_loss=criterion.loss.data, epoch=i)
            else:
                helper.log("Epoch {}/{} training loss = {:.5f}".format(i + 1, args.epochs, criterion.loss.data))

    def _infer(self, batch, training=True, use_negative=True):
        args = self._args
        if use_negative:
            neg = torch.LongTensor(batch.negative + 1).to(device)
            neg_nh, neg_msk = batch.negative_neighborhood.to(device), batch.negative_mask.to(device)
        else:
            neg, neg_nh, neg_msk = None, None, None

        self.model(source=torch.LongTensor(batch.source + 1).to(device),
                   target=torch.LongTensor(batch.target + 1).to(device), negative=neg,
                   source_neighborhood=batch.source_neighborhood.to(device),
                   target_neighborhood=batch.target_neighborhood.to(device),
                   negative_neighborhood=neg_nh, source_mask=batch.source_mask.to(device),
                   target_mask=batch.target_mask.to(device), negative_mask=neg_msk, training=training)

        return (to_cpu_tensor(self.model.source_emb), to_cpu_tensor(self.model.target_emb),
                to_cpu_tensor(self.model.source_rep), to_cpu_tensor(self.model.target_rep))

    def infer_embeddings(self, agg=lambda l: np.mean(l, axis=0)):
        #         def populate_embedding(source_nodes, target_nodes, source_emb, target_emb):
        #             def apply(idx_emb, idx):
        #                 node_id = self.data.index_to_node[idx]
        #                 if node_id in self.context_embedding:
        #                     self.context_embedding[node_id].append(idx_emb)
        #                 else:
        #                     self.context_embedding[node_id] = [idx_emb]

        def populate_embedding(source_nodes, target_nodes, source_emb, target_emb, emb_dict, context=True):
            def apply(idx_emb, idx):
                if context:
                    if idx in emb_dict:
                        emb_dict[idx].append(idx_emb)
                    else:
                        emb_dict[idx] = [idx_emb]
                else:
                    if idx not in emb_dict:
                        emb_dict[idx] = idx_emb

            for i in range(len(source_nodes)):
                apply(source_emb[i], source_nodes[i])
                apply(target_emb[i], target_nodes[i])

        for batch in self.data.train_batches + self.data.dev_batches:
            src_emb, trg_emb, src_rep, trg_rep = self._infer(batch=batch, use_negative=False, training=False)
            self.attention_data['source'].extend(batch.source)
            self.attention_data['target'].extend(batch.target)
            self.attention_data['source_neighborhood'].extend(to_cpu_tensor(batch.source_neighborhood - 1))
            self.attention_data['target_neighborhood'].extend(to_cpu_tensor(batch.target_neighborhood - 1))
            self.attention_data['source_attentions'].extend(to_cpu_tensor(self.model.source_attention_vec.squeeze()))
            self.attention_data['target_attentions'].extend(to_cpu_tensor(self.model.target_attention_vec.squeeze()))
            populate_embedding(batch.source, batch.target, src_rep, trg_rep, emb_dict=self.context_embedding)
            populate_embedding(batch.source, batch.target, src_emb, trg_emb, emb_dict=self.global_embedding,
                               context=False)

        self.attention_data['source'] = np.array(self.attention_data['source'])
        self.attention_data['target'] = np.array(self.attention_data['target'])
        self.attention_data['source_neighborhood'] = np.array(self.attention_data['source_neighborhood'])
        self.attention_data['target_neighborhood'] = np.array(self.attention_data['target_neighborhood'])
        self.attention_data['source_attentions'] = np.array(self.attention_data['source_attentions'])
        self.attention_data['target_attentions'] = np.array(self.attention_data['target_attentions'])

    def save_embeddings(self):
        args = self._args
        if args.output_dir != '':
            suffix = '' if args.tr_rate == 1 else f'_{str(int(args.tr_rate * 100))}'
            path = os.path.join(args.output_dir, f'goat_context{suffix}.emb')
            helper.log(f'Saving context embedding to {path}')
            with open(path, 'w') as f:
                for node in self.context_embedding:
                    for emb in self.context_embedding[node]:
                        output = '{} {}\n'.format(node, ' '.join(str(val) for val in emb))
                        f.write(output)

            path = os.path.join(args.output_dir, f'goat_global{suffix}.emb')
            helper.log(f'Saving global embedding to {path}')
            with open(path, 'w') as f:
                for node in self.global_embedding:
                    output = '{} {}\n'.format(node, ' '.join(str(val) for val in self.global_embedding[node]))
                    f.write(output)

            path = os.path.join(args.output_dir, f'attention_weights')
            helper.log(f'Saving relevant data for analysing attention weights to {path}.npz')
            np.savez_compressed(
                path, source=self.attention_data['source'], target=self.attention_data['target'],
                source_neighborhood=self.attention_data['source_neighborhood'],
                target_neighborhood=self.attention_data['target_neighborhood'],
                source_attentions=self.attention_data['source_attentions'],
                target_attentions=self.attention_data['target_attentions'])


def main(args, train_graph=None, test_graph=None):
    helper.VERBOSE = False if args.verbose == 0 else True

    wrapper = GoatWrapper(args, train_graph=train_graph, test_graph=test_graph)
    # import datetime as dt
    # start = dt.datetime.now()
    wrapper.train()
    # end = dt.datetime.now()
    # delta = end - start
    # print(delta.seconds)
    if args.output_dir != '':
        wrapper.infer_embeddings()
        wrapper.save_embeddings()


if __name__ == '__main__':
    main(helper.parse_args())