"""
Author: Zekarias Tilahun Kefato <zekarias@kth.se>


This module implements the core of the GOAT model.
GOAT is an acronym for the paper titled

    Gossip and Attend: A Context-sensitive Graph Representation Learning

"""

from utils import parse_train_args, Trainer
from data import Data, InputDataset, DataLoader
from model import Goat


if __name__ == '__main__':
    args = parse_train_args()
    
    data_ = Data(args=args)
    
    train_dataset = InputDataset(input_data=data_, num_neg=args.num_neg, partition="train")
    dev_dataset = InputDataset(input_data=data_, num_neg=args.num_neg, partition="dev")

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch)
    dev_loader = DataLoader(dataset=train_dataset, batch_size=args.batch)

    goat = Goat(num_nodes=data_.num_nodes, dim=args.dim, dropout=args.dropout)
    
    trainer = Trainer(config=args)
    trainer.fit(model=goat, train_loader=train_loader, dev_loader=dev_loader)
