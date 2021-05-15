# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
import paddorch as torch
import paddorch.nn as nn
import numpy as np
import logging
from data import load_data, read_relation_subsets, gen_rel_subset_feature
from model import SIGN, WeightedAggregator
from utils import get_n_params, get_evaluator, train, test
import pickle

def preprocess_features(g, rel_subsets, args, device):
    # pre-process heterogeneous graph g to generate neighbor-averaged features
    # for each relation subsets
    num_paper, feat_size = g.nodes["paper"].data["feat"].shape
    new_feats = [np.zeros((num_paper, len(rel_subsets), feat_size)) for _ in range(args.R + 1)]
    print("Start generating features for each sub-metagraph:")
    for subset_id, subset in enumerate(rel_subsets):
        print(subset)
        feats = gen_rel_subset_feature(g, subset, args, device)
        for i in range(args.R + 1):
            feat = feats[i]
            new_feats[i][:feat.shape[0], subset_id, :] = feat
        feats = None
    return new_feats


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.manual_seed(args.seed)

    # Load dataset
    data = load_data(device, args)
    print("loaded data")
    g, labels, num_classes, train_nid, val_nid, test_nid = data
    evaluator = get_evaluator(args.dataset)
    print("loaded evaluator")
    # Preprocess neighbor-averaged features over sampled relation subgraphs
    rel_subsets = read_relation_subsets(args.use_relation_subsets)
    args.model_name = "{}_nh{}_R{}_fl{}_drop{}_idrop{}_ue{}_seed{}_lr{}_wd{}".format(
        args.dataset,
        args.num_hidden,
        args.R,
        args.ff_layer,
        args.dropout,
        args.input_dropout,
        args.use_emb,
        args.seed,
        args.lr,
        args.weight_decay
    )
    args.cache_path = ".cache/"
    args.ckpt_path = "checkpoints/"
    args.model_folder = args.ckpt_path + "/" + args.model_name
    import os
    os.makedirs(args.cache_path, exist_ok=True)
    os.makedirs(args.model_folder, exist_ok=True)
    # Preprocess neighbor-averaged features over sampled relation subgraphs
    cache_fn = args.cache_path + "/" + args.model_name + "_feats.pkl"
    if os.path.isfile(cache_fn):
        feats = pickle.load(open(cache_fn, 'rb'))
        print("successfully loaded preprocess_features cache data from", cache_fn)
    else:
        with torch.no_grad():
            feats = preprocess_features(g, rel_subsets, args, device)
            print("Done preprocessing")
        pickle.dump(feats, open(cache_fn, 'wb'))

    # Release the graph since we are not going to use it later
    g = None

    # Set up logging
    logging.basicConfig(format='[%(levelname)s] %(message)s',
                        level=logging.INFO)
    logging.info(str(args))

    _, num_feats, in_feats = feats[0].shape
    logging.info(f"new input size: {num_feats} {in_feats}")

    # Create model
    num_hops =  args.R + 1  # include self feature hop 0
    model = nn.Sequential(
        WeightedAggregator(num_feats, in_feats, num_hops),
        SIGN(in_feats, args.num_hidden, num_classes, num_hops,
             args.ff_layer, args.dropout, args.input_dropout)
    )
    logging.info("# Params: {}".format(get_n_params(model)))
    model.to(device)

    if len(labels.shape) == 1:
        # single label multi-class
        loss_fcn = nn.NLLLoss()
    else:
        # multi-label multi-class
        loss_fcn = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Start training
    best_epoch = 0
    best_val = 0
    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        train(model, feats, labels, train_nid, loss_fcn, optimizer, args.batch_size)

        if epoch % args.eval_every == 0:
            with torch.no_grad():
                train_res, val_res, test_res = test(
                    model, feats, labels, train_nid, val_nid, test_nid, evaluator, args.eval_batch_size)
            end = time.time()
            val_acc = val_res[0]
            log = "Epoch {}, Times(s): {:.4f}".format(epoch, end - start)
            if args.dataset.startswith("oag"):
                log += ", NDCG: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(train_res[0], val_res[0], test_res[0])
                log += ", MRR: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(train_res[1], val_res[1], test_res[1])
            else:
                log += ", Accuracy: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(train_res[0], val_res[0], test_res[0])
            logging.info(log)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch

    logging.info("Best Epoch {}, Val {:.4f}".format(best_epoch, best_val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neighbor-Averaging over Relation Subgraphs")
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--num-hidden", type=int, default=256)
    parser.add_argument("--R", type=int, default=2,
                        help="number of hops")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="oag")
    parser.add_argument("--data-dir", type=str, default=None, help="path to dataset, only used for OAG")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--eval-batch-size", type=int, default=250000,
                        help="evaluation batch size, -1 for full batch")
    parser.add_argument("--ff-layer", type=int, default=2,
                        help="number of feed-forward layers")
    parser.add_argument("--input-dropout", action="store_true")
    parser.add_argument("--use-emb", required=True, type=str)
    parser.add_argument("--use-relation-subsets", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0 )
    parser.add_argument("--cpu-preprocess", action="store_true",
                        help="Preprocess on CPU")
    args = parser.parse_args()

    print(args)
    main(args)
