from collections import namedtuple

from data import compile_training_data

import torch

import numpy as np

import evaluate
import helper
import model

import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GoatWrapper:

    def __init__(self, args, train_graph=None, test_graph=None):
        self._args = args
        self.data = compile_training_data(args, train_graph=train_graph, test_graph=test_graph)
        self.loss_fun = model.LogLoss
        self.model = None
        self.context_embedding = {}
        self.global_embedding = {}
        self.attention_data = {
            'source': [], 'target': [], 'source_neighborhood': [], 'target_neighborhood':[], 
            'source_attentions': [], 'target_attentions': []}

    def _validate(self):
        losses = []
        aucs = []
        for batch in self.data.dev_batches:
            fb = format_batch(batch=batch, use_negative=True)
            _, _, source_rep, target_rep = self._infer(batch=fb)
            val_loss = self.loss_fun(self.model)
            link_probs = evaluate.compute_link_probabilities(u_embed=source_rep, v_embed=target_rep)
            cur_results = evaluate.compute_results(link_probabilities=link_probs, eval_metrics={'auc'})
            
            losses.append(as_numpy_array(val_loss.loss))
            aucs.append(cur_results['auc'])
            
        return losses, aucs

    def train(self):
        args = self._args
        self.model = model.GOAT(num_nodes=self.data.num_nodes, emb_dim=args.dim)
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        total_batches_ = self.data.total_batches

        helper.log(f'Total batches per epoch {total_batches_}')
        train_loss = 0
        for i in range(args.epochs):
            for batch in self.data.train_batches:
                fb = format_batch(batch=batch, use_negative=True)
                self._infer(fb)
                criterion = self.loss_fun(self.model)
                optimizer.zero_grad()
                criterion.loss.backward()
                optimizer.step()
                train_loss = criterion.loss.data

            if args.dev_rate > 0:
                dev_losses, dev_aucs = self._validate()
                helper.log(
                    'Epoch: {}/{} training loss: {:.5f} validation loss: {:.5f} validation AUC: {:.5f}'.format(
                        i + 1, args.epochs, train_loss, np.mean(dev_losses), np.mean(dev_aucs)))
            else:
                helper.log("Epoch {}/{} training loss = {:.5f}".format(i + 1, args.epochs, train_loss))

    def _infer(self, batch, training=True):
        self.model(source=batch.source, target=batch.target, negative=batch.negative,
                   source_neighborhood=batch.source_nh,
                   target_neighborhood=batch.target_nh,
                   negative_neighborhood=batch.negative_nh,
                   source_mask=batch.source_mask,
                   target_mask=batch.target_mask,
                   negative_mask=batch.negative_mask, training=training)
        
        return (as_numpy_array(self.model.source_emb), as_numpy_array(self.model.target_emb),
                as_numpy_array(self.model.source_rep), as_numpy_array(self.model.target_rep))

    def infer_embeddings(self):

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

        def embed(batches):
            for batch in batches:
                fb = format_batch(batch=batch, use_negative=False)
                src_glb_emb, trg_glb_emb, src_ctx_rep, trg_ctx_rep = self._infer(batch=fb, training=False)
                self.attention_data['source'].extend(fb.source)
                self.attention_data['target'].extend(fb.target)
                self.attention_data['source_neighborhood'].extend(as_numpy_array(fb.source_nh - 1))
                self.attention_data['target_neighborhood'].extend(as_numpy_array(fb.target_nh - 1))
                self.attention_data['source_attentions'].extend(as_numpy_array(
                    self.model.source_attention_vec.squeeze()))
                self.attention_data['target_attentions'].extend(as_numpy_array(
                    self.model.target_attention_vec.squeeze()))
                populate_embedding(fb.source, fb.target, src_ctx_rep, trg_ctx_rep, emb_dict=self.context_embedding)
                populate_embedding(fb.source, fb.target, src_glb_emb, trg_glb_emb, emb_dict=self.global_embedding,
                                   context=False)

        embed(self.data.train_batches)
        embed(self.data.dev_batches)

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
    wrapper.train()
    if args.output_dir != '':
        wrapper.infer_embeddings()
        wrapper.save_embeddings()


# noinspection PyArgumentList
def format_batch(batch, use_negative):
    formatted_batch = {}
    if isinstance(batch, tuple):
        formatted_batch["source"] = torch.LongTensor(batch.source + 1).to(device)
        formatted_batch["target"] = torch.LongTensor(batch.target + 1).to(device)
        formatted_batch["source_nh"] = batch.source_neighborhood.to(device)
        formatted_batch["target_nh"] = batch.target_neighborhood.to(device)
        formatted_batch["source_mask"] = batch.source_mask.to(device)
        formatted_batch["target_mask"] = batch.target_mask.to(device)
        if use_negative:
            formatted_batch["negative"] = torch.LongTensor(batch.negative + 1).to(device)
            formatted_batch["negative_nh"] = batch.negative_neighborhood.to(device)
            formatted_batch["negative_mask"] = batch.negative_mask.to(device)
        else:
            formatted_batch["negative"], formatted_batch["negative_nh"] = None, None
            formatted_batch["negative_mask"] = None
    else:
        formatted_batch["source"] = (batch[0] + 1).to(device)
        formatted_batch["target"] = (batch[1] + 1).to(device)
        formatted_batch["source_nh"] = batch[3].to(device)
        formatted_batch["target_nh"] = batch[4].to(device)
        formatted_batch["source_mask"] = batch[6].to(device)
        formatted_batch["target_mask"] = batch[7].to(device)
        if use_negative:
            formatted_batch["negative"] = (batch[2] + 1).to(device)
            formatted_batch["negative_nh"], formatted_batch["negative_mask"] = batch[5].to(device), batch[8].to(device)
        else:
            formatted_batch["negative"], formatted_batch["negative_nh"] = None, None
            formatted_batch["negative_mask"] = None
    return namedtuple("FormattedBatch", formatted_batch.keys())(*formatted_batch.values())


def as_numpy_array(tensor):
    if device == 'cpu':
        return tensor.data.numpy()
    return tensor.cpu().data.numpy()


if __name__ == '__main__':
    main(helper.parse_args())