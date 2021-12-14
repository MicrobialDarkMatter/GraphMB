import time
import pdb
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from dgl.nn.pytorch.conv import SAGEConv, GATConv
import dgl.function as fn
import tqdm
import numpy as np
import copy
import sklearn

from graphmb.graph_functions import cluster_eval, set_seed

# Based on this implemention: https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_sampling_unsupervised.py


class MultiLayerNeighborWeightedSampler(dgl.dataloading.MultiLayerNeighborSampler):
    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout, replace=self.replace, prob="weight")
        return frontier


class NegativeSampler(object):
    def __init__(self, g, k, neg_share=False):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n * self.k, replacement=True)
        src = src.repeat_interleave(self.k)
        return src, dst


class NegativeSamplerWeight(object):
    """Samples negatives according to the inverse of edge weight"""

    def __init__(self, g, k, neg_share=False):
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, dst = g.find_edges(eids)
        new_dst = []
        new_src = []
        possible_dst = set(dst)
        new_src = [src_node for src_node in src if len(g.out_edges(src_node)[1]) < 2]
        new_dst = random.sample(possible_dst, k=len(new_src))
        # print("adding", len(new_src), "more edges")
        src = src.repeat_interleave(self.k)
        expand_g = dgl.remove_self_loop(dgl.add_edges(g, new_src, new_dst))
        expand_g.edata["weight"] = 1 / expand_g.edata["weight"]

        samples_edges = dgl.sampling.sample_neighbors(
            expand_g, src, self.k, prob="weight", replace=True, edge_dir="out"
        )
        src = samples_edges.edges()[0]
        dst = samples_edges.edges()[1]
        # src = src.repeat_interleave(self.k)
        return src, dst


class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph, weights=True):
        with pos_graph.local_scope():
            pos_graph.ndata["h"] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            pos_score = pos_graph.edata["score"]
        with neg_graph.local_scope():
            neg_graph.ndata["h"] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            neg_score = neg_graph.edata["score"]

        score = torch.cat([pos_score, neg_score])
        pos_label = torch.ones_like(pos_score)
        if weights:
            pos_weights = pos_graph.edata["weight"] / max(pos_graph.edata["weight"])
            pos_weights = pos_weights.unsqueeze(-1)
        else:
            pos_weights = torch.ones_like(pos_score)
        neg_label = torch.zeros_like(neg_score)
        neg_weights = torch.ones_like(neg_score)
        all_weights = torch.cat([pos_weights, neg_weights])
        label = torch.cat([pos_label, neg_label]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float(), weight=all_weights)
        return loss


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, agg="mean"):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout, agg)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, agg):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        # self.batchnorm = nn.BatchNorm1d(n_hidden)
        if n_layers > 1:
            self.layers.append(SAGEConv(in_feats, n_hidden, agg, feat_drop=dropout))

            for i in range(1, n_layers - 1):
                self.layers.append(SAGEConv(n_hidden, n_hidden, agg, feat_drop=dropout))
                # self.layers.append(nn.BatchNorm1d(n_hidden))
            self.layers.append(SAGEConv(n_hidden, n_classes, agg))
        else:
            self.layers.append(SAGEConv(in_feats, n_classes, agg, feat_drop=dropout))
        if agg == "gcn":
            breakpoint()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x, edge_weight=None):
        h = x
        for il, (layer, block) in enumerate(zip(self.layers, blocks)):
            # weights = F.softmax(block.edata["weight"])
            # weights = block.edata["weight"] / max(block.edata["weight"])
            # h = layer(block, h, edge_weight=weights)
            h = layer(block, h)
            if il != len(self.layers) - 1:
                h = self.activation(h)
        return h

    def inference(self, g, x, device, batch_size, num_workers, use_weights=False):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for il, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden if il != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.num_nodes()).to(g.device),
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            )

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)

                if use_weights:
                    weights = block.edata["weight"] / max(block.edata["weight"])
                    h = layer(block, h, edge_weight=weights)
                else:
                    h = layer(block, h)
                if il != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y


def train_graphsage(
    dataset,
    model,
    batch_size,
    fan_out,
    num_negs,
    neg_share,
    lr,
    num_epochs,
    num_workers=0,
    print_interval=3,
    device="cpu",
    cluster_features=True,
    clusteringalgo="kmeans",
    k=1,
    logger=None,
    loss_weights=False,
    sample_weights=False,
    epsilon=0.1,
):

    nfeat = dataset.graph.ndata.pop("feat")
    model = model.to(device)
    # Create PyTorch DataLoader for constructing blocks
    n_edges = dataset.graph.num_edges()
    train_seeds = torch.arange(n_edges)
    set_seed()

    # Create samplers
    if not sample_weights:
        neg_sampler = NegativeSampler(dataset.graph, num_negs, neg_share)
        sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in fan_out.split(",")])
    else:
        neg_sampler = NegativeSamplerWeight(dataset.graph, num_negs, neg_share)
        sampler = MultiLayerNeighborWeightedSampler([int(fanout) for fanout in fan_out.split(",")])

    if batch_size == 0:
        batch_size = len(train_seeds)

    dataloader = dgl.dataloading.EdgeDataLoader(
        dataset.graph,
        train_seeds,
        sampler,
        exclude="reverse_id",
        reverse_eids=torch.cat([torch.arange(n_edges // 2, n_edges), torch.arange(0, n_edges // 2)]).to(train_seeds),
        negative_sampler=neg_sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    loss_fcn = CrossEntropyLoss()
    optimizer_sage = optim.Adam(model.parameters(), lr=lr)
    # Training loop
    avg = 0
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    best_hq = 0
    best_hq_epoch = 0
    total_steps = 0
    losses = []
    for epoch in range(num_epochs):
        tic = time.time()
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
            batch_inputs = nfeat[input_nodes].to(device)
            d_step = time.time()
            set_seed()
            model.train()
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            blocks = [block.int().to(device) for block in blocks]
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, pos_graph, neg_graph, weights=loss_weights)

            optimizer_sage.zero_grad()
            loss.backward()
            optimizer_sage.step()

            t = time.time()
            pos_edges = pos_graph.num_edges()
            neg_edges = neg_graph.num_edges()
            iter_pos.append(pos_edges / (t - tic_step))
            iter_neg.append(neg_edges / (t - tic_step))
            iter_d.append(d_step - tic_step)
            iter_t.append(t - d_step)
            if total_steps % print_interval == 0:
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                logger.info(
                    "Epoch {:05d} | Step {:05d} | N samples {} | Loss {:.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB".format(
                        epoch,
                        step,
                        len(input_nodes),
                        loss.item(),
                        np.mean(iter_pos[3:]),
                        np.mean(iter_neg[3:]),
                        np.mean(iter_d[3:]),
                        np.mean(iter_t[3:]),
                        gpu_mem_alloc,
                    )
                )
            tic_step = time.time()
            total_steps += 1

        losses.append(loss.item())
        # early stopping
        if (
            epsilon is not None
            and len(losses) > 3
            and (losses[-2] - losses[-1]) < epsilon
            and (losses[-3] - losses[-2]) < epsilon
        ):
            logger.info("Early stopping {}".format(str(losses[-5:])))
            break

        model.eval()
        encoded = model.inference(dataset.graph, nfeat, device, batch_size, num_workers)

        if cluster_features:
            encoded = torch.cat((encoded, nfeat), axis=1)
        if dataset.ref_marker_sets is not None:
            best_hq, best_hq_epoch, kmeans_loss, clusters = cluster_eval(
                model=model,
                dataset=dataset,
                logits=encoded,
                clustering=clusteringalgo,
                k=k,
                loss=loss,
                best_hq=best_hq,
                best_hq_epoch=best_hq_epoch,
                epoch=epoch,
                device=device,
                clusteringloss=False,
                logger=logger,
            )

            # compare clusters
            new_assignments = np.zeros(len(dataset.node_names))
            for i, cluster in enumerate(clusters):
                for contig in clusters[cluster]:
                    new_assignments[dataset.contig_names.index(contig)] = i

            old_assignments = new_assignments.copy()
        else:
            logger.info(
                "Epoch {:05d} | Best HQ: {} | Best epoch {} | Total loss {:.4f}".format(
                    epoch,
                    best_hq,
                    best_hq_epoch,
                    loss.detach(),
                )
            )
        toc = time.time()
        if epoch >= 5:
            avg += toc - tic
        encoded = encoded.cpu().detach().numpy()

    last_train_embs = encoded
    last_model = model
    logger.info("saving last model")
    torch.save(last_model.state_dict(), os.path.join(dataset.assembly, "last_model_hq.pkl"))
    logger.info("Avg epoch time: {}".format(avg / (epoch - 4)))
    model.eval()
    logger.info(f"Best HQ {best_hq} epoch, {best_hq_epoch}")
    if total_steps > 0 and dataset.ref_marker_sets is not None:
        logger.info("loading best model")
        best_model = copy.deepcopy(model)
        try:
            best_model.load_state_dict(torch.load(os.path.join(dataset.assembly, "best_model_hq.pkl")))
        except RuntimeError:
            pdb.set_trace()
    set_seed()
    print("running best model again")
    best_train_embs = best_model.inference(dataset.graph, nfeat, device, batch_size, num_workers)
    best_train_embs = best_train_embs.detach()
    if cluster_features:
        best_train_embs = torch.cat((best_train_embs, nfeat), axis=1).detach()
    best_hq, best_hq_epoch, kmeans_loss, clusters = cluster_eval(
        model=best_model,
        dataset=dataset,
        logits=best_train_embs,
        clustering=clusteringalgo,
        k=k,
        loss=loss,
        best_hq=best_hq,
        best_hq_epoch=best_hq_epoch,
        epoch=epoch,
        device=device,
        clusteringloss=False,
        logger=logger,
    )
    return best_train_embs, best_model, last_train_embs, last_model
