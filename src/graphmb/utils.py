from pathlib import Path
import time
import os
import sys
import math
import pdb
import itertools
from collections import Counter
import networkx as nx
import numpy as np
from tqdm import tqdm
import datetime
import operator
import scipy
import logging
# import dgl
import random
from sklearn.cluster import KMeans
#import tensorflow as tf

SEED = 0

def set_seed(seed=0):
    if "dgl" in sys.modules:
        import dgl
        print("setting dgl seed")
        dgl.random.seed(seed)
    if "torch" in sys.modules:
        import torch
        print("setting torch seed")
        torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if "tensorflow" in sys.modules:
        import tensorflow
        print("setting tf seed")
        tensorflow.random.set_seed(seed)



class Read:
    def __init__(self, readid, species=None):
        self.readid = readid
        self.species = species
        self.mappings = set()


class ReadMapping:
    def __init__(self, readid, bitflag, contigname, pos, mapq, seq):
        self.readid = readid
        self.bitflag = bitflag
        self.contigname = contigname
        self.pos = pos
        self.mapq = mapq


def get_cluster_mask(quick, dataset):
    if quick and dataset.contig_markers is not None:
        #connected_marker_nodes = filter_disconnected(dataset.adj_matrix, dataset.node_names, dataset.contig_markers)
        nodes_with_markers = [
            i
            for i, n in enumerate(dataset.node_names)
            if n in dataset.contig_markers and len(dataset.contig_markers[n]) > 0
        ]
        print("eval cluster with ", len(nodes_with_markers), "contigds with markers")
        cluster_mask = [n in nodes_with_markers for n in range(len(dataset.node_names))]
    else:
        cluster_mask = [True] * len(dataset.node_names)
    return cluster_mask

def save_model(args, epoch, th, th_vae):
    if th_vae is not None:
        # save encoder and decoder
        th_vae.encoder.save(os.path.join(args.outdir, args.outname + "_best_encoder"))
        th_vae.decoder.save(os.path.join(args.outdir, args.outname + "_best_decoder"))
    if th is not None:
        th.gnn_model.save(os.path.join(args.outdir, args.outname + "_best_gnn"))


def run_clustering(X, node_names, clustering_algo, cuda, k=0, tsne=False):
    
    if clustering_algo == "vamb":
        from graphmb.vamb_clustering import cluster as vamb_cluster
        starttime = datetime.datetime.now()
        X = X.astype(np.float32)
        cluster_to_contig = {
            i: c for (i, (n, c)) in enumerate(vamb_cluster(X, node_names, cuda=cuda))
        }
        clustering_time = datetime.datetime.now()
        #print("clustering time", clustering_time-starttime)
        contig_to_bin = {}
        #for b in cluster_to_contig:
        #    for contig in cluster_to_contig[b]:
        #        contig_to_bin[contig] = b
        for k, v in cluster_to_contig.items():
            contig_to_bin.update({n: k for n in v})
        labels = np.array([contig_to_bin[n] for n in node_names])
        # very slow code:
        cluster_centroids = None
        if tsne:
            cluster_to_embs = {
                c: np.array([X[i] for i, n in enumerate(node_names) if n in cluster_to_contig[c]])
                for c in cluster_to_contig
            }
            cluster_centroids = np.array([cluster_to_embs[c].mean(0) for c in cluster_to_contig])
        processing_time = datetime.datetime.now()
        #print("processing time",  processing_time - clustering_time)
    elif clustering_algo == "kmeansbatch":
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=2048, verbose=0) #, init=seed_matrix)
        labels = kmeans.fit_predict(X)
        contig_to_bin = {node_names[i]: labels[i] for i in range(len(node_names))}
        cluster_to_contig = {i: [] for i in range(k)}
        for i in range(len(node_names)):
            cluster_to_contig[labels[i]].append(node_names[i])
        #cluster_centroids = kmeans.cluster_centers_
    elif clustering_algo == "kmeansgpu":
        pass
    elif clustering_algo == "kmedoids":
        import kmedoids
        breakpoint()
        # TODO do this on gpu if avail
        D = np.sum((X[:,None]-X[None])**2, axis=-1)
        # TODO find best k
        km = kmedoids.KMedoids(20, method='fasterpam')
        cluster_labels = km.fit_predict(D).astype(np.int64)
    elif clustering_algo == "kmeans":
        clf = KMeans(k, random_state=1234)
        labels = clf.fit_predict(X)
        contig_to_bin = {node_names[i]: labels[i] for i in range(len(node_names))}
        cluster_to_contig = {i: [] for i in range(k)}
        for i in range(len(node_names)):
            cluster_to_contig[labels[i]].append(node_names[i])
        cluster_centroids = None
    return cluster_to_contig, contig_to_bin, labels, cluster_centroids

def filter_disconnected(adj, node_names, markers):
    # get idx of nodes that are connected or have at least one marker
    graph = nx.convert_matrix.from_scipy_sparse_matrix(adj, edge_attribute="weight")
    # breakpoint()
    nodes_to_remove = set()
    for n1 in graph.nodes:
        if len(list(graph.neighbors(n1))) == 0 or (
            node_names[n1] not in markers or len(markers[node_names[n1]]) == 0
        ):
            nodes_to_remove.add(n1)

    graph.remove_nodes_from(list(nodes_to_remove))
    assert len(graph.nodes()) == (len(node_names)-len(nodes_to_remove))
    print(f"{len(nodes_to_remove)} nodes without edges nor markers, keeping {len(graph.nodes())} nodes")
    return set(graph.nodes())

def run_model_vgae(dataset, args, logger, nrun):
    node_names = np.array(dataset.node_names)
    RESULT_EVERY = args.evalepochs
    hidden_gnn = args.hidden_gnn
    hidden_vae = args.hidden_vae
    output_dim_gnn = args.embsize_gnn
    output_dim_vae = args.embsize_vae
    epochs = args.epoch
    lr_vae = args.lr_vae
    lr_gnn = args.lr_gnn
    nlayers_gnn = args.layers_gnn
    clustering = args.clusteringalgo
    k = args.kclusters
    use_edge_weights = True
    cluster_markers_only = args.quick
    decay = 0.5 ** (2.0 / epochs)
    concat_features = args.concat_features

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(args.outdir, 'logs/' + args.outname + current_time + '/train')
    summary_writer = tf.summary.create_file_writer(train_log_dir)
    print("logging to tensorboard")
    tb_handler = TensorboardLogger(summary_writer, runname=args.outname + current_time)
    logger.addHandler(tb_handler)
    #tf.summary.trace_on(graph=True)

    logger.info("******* Running model: VGAE **********")
    logger.info("***** using edge weights: {} ******".format(use_edge_weights))
    logger.info("***** concat features: {} *****".format(concat_features))
    logger.info("***** cluster markers only: {} *****".format(cluster_markers_only))
    logger.info("***** threshold adj matrix: {} *****".format(args.binarize))
    logger.info("***** self edges only: {} *****".format(args.noedges))
    logger.info("***** Using raw kmer+abund features: {}".format(args.rawfeatures))
    tf.config.experimental_run_functions_eagerly(True)


    X, adj, cluster_mask, neg_pair_idx, pos_pair_idx = prepare_data_for_gnn(
            dataset, use_edge_weights, cluster_markers_only, use_raw=True,
            binarize=args.binarize, remove_edges=args.noedges)
    logger.info("***** SCG neg pairs: {}".format(neg_pair_idx.shape))
    logger.info("***** input features dimension: {}".format(X[cluster_mask].shape))
    # pre train clustering
    if not args.skip_preclustering:
        cluster_labels, stats, _, hq_bins = compute_clusters_and_stats(
                    X[cluster_mask], node_names[cluster_mask],
                    dataset, clustering=clustering, k=k, tsne=args.tsne,
                    amber=(args.labels is not None and "amber" in args.labels),
                    unresolved=True, cuda=args.cuda, 
                )
        logger.info(f">>> Pre train stats: {str(stats)}")


    model = VGAE(X.shape, hidden_dim1=hidden_gnn, hidden_dim2=output_dim_gnn, dropout=0.1,
             l2_reg=1e-5, embeddings=X, freeze_embeddings=True, lr=lr_gnn)
    X_train = np.arange(len(X))[:,None].astype(np.int64)
    A_train = tf.sparse.to_dense(adj)
    labels = dataset.adj_matrix.toarray()
    pos_weight = (adj.shape[0] * adj.shape[0] - tf.sparse.reduce_sum(adj)) / tf.sparse.reduce_sum(adj)

    norm = adj.shape[0] * adj.shape[0] / ((adj.shape[0] * adj.shape[0] - tf.sparse.reduce_sum(adj)) * 2)

    pbar_epoch = tqdm(range(epochs), disable=args.quiet, position=0)
    decay = 0.5**(2./10000)
    scores = []
    best_hq = 0
    batch_size = args.batchsize
    if batch_size == 0:
        batch_size = adj.shape[0]
    train_idx = list(range(adj.shape[0]))
    for e in pbar_epoch:
        np.random.shuffle(train_idx)
        n_batches = len(train_idx)//batch_size
        pbar_vaebatch = tqdm(range(n_batches), disable=(args.quiet or batch_size == len(train_idx) or n_batches < 100), position=1, ascii=' =')
        loss = 0
        for b in pbar_vaebatch:
            batch_idx = train_idx[b*batch_size:(b+1)*batch_size]
            loss += model.train_step(X_train, A_train, labels, pos_weight, norm, batch_idx)
        pbar_epoch.set_description(f'{loss:.3f}')
        model.optimizer.learning_rate = model.optimizer.learning_rate*decay
        gpu_mem_alloc = tf.config.experimental.get_memory_usage('GPU:0') / 1000000 if args.cuda else 0
        if (e + 1) % RESULT_EVERY == 0: # and e >= int(epochs/2):
            _, embs, _, _, _ = model((X_train, A_train), training=False)
            node_new_features = embs.numpy()

            best_hq, best_embs, best_epoch, scores, cluster_labels = eval_epoch(logger, summary_writer, node_new_features,
                                                                cluster_mask, e, args, dataset, e, scores,
                                                                best_hq, best_embs, best_epoch)

            if args.quiet:
                logger.info(f"--- EPOCH {e:d} ---")
                logger.info(f"[VGAE {nlayers_gnn}l] L={loss:.3f}  HQ={stats['hq']}  BestHQ={best_hq} Best Epoch={best_epoch} Max GPU MB={gpu_mem_alloc:.1f}")
                logger.info(str(stats))


    _, embs, _, _, _ = model((X_train, A_train), training=False)
    embs = embs.numpy()


