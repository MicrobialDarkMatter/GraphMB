from sklearn.cluster import KMeans
import sys
import os
import random
from tqdm import tqdm
import itertools
import logging
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from rich.console import Console
from rich.table import Table
from scipy.sparse import csr_matrix, diags

from graphmb.models import SAGE, SAGELAF, GCN, GCNLAF, GAT, GATLAF, TH, TrainHelperVAE, VAEDecoder, VAEEncoder
from graph_functions import set_seed, run_tsne, plot_embs
from graphmb.evaluate import calculate_overall_prf

name_to_model = {"SAGE": SAGE, "SAGELAF": SAGELAF, "GCN": GCN, "GCNLAF": GCNLAF, "GAT": GAT, "GATLAF": GATLAF}


def normalize_adj_sparse(A):
    # https://github.com/tkipf/gcn/blob/master/gcn/utils.py
    A.setdiag(1)
    rowsum = np.array(A.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = diags(d_inv_sqrt)
    return A.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def completeness(reference_markers, genes):
    numerator = 0.0
    for marker_set in reference_markers:
        common = marker_set & genes
        if len(marker_set) > 0:
            numerator += len(common) / len(marker_set)
    return 100 * (numerator / len(reference_markers))


def contamination(reference_markers, genes):
    numerator = 0.0
    for i, marker_set in enumerate(reference_markers):
        inner_total = 0.0
        for gene in marker_set:
            if gene in genes and genes[gene] > 0:
                inner_total += genes[gene] - 1.0
        if len(marker_set) > 0:
            numerator += inner_total / len(marker_set)
    return 100.0 * (numerator / len(reference_markers))


def compute_cluster_score(reference_markers, contig_genes, node_names, node_labels):
    labels_to_nodes = {i: node_names[node_labels == i].tolist() for i in np.unique(node_labels)}
    results = {}
    for label in labels_to_nodes:
        genes = {}
        for node_name in labels_to_nodes[label]:
            if node_name not in contig_genes:
                # print("missing", node_name)
                continue
            for gene in contig_genes[node_name]:
                if gene not in genes:
                    genes[gene] = 0
                genes[gene] += contig_genes[node_name][gene]

        comp = completeness(reference_markers, set(genes.keys()))
        cont = contamination(reference_markers, genes)
        results[label] = {"comp": comp, "cont": cont, "genes": genes}
    return results


def compute_hq(reference_markers, contig_genes, node_names, node_labels, comp_th=90, cont_th=5):
    cluster_stats = compute_cluster_score(reference_markers, contig_genes, node_names, node_labels)
    hq = 0
    positive_clusters = []
    for label in cluster_stats:
        if cluster_stats[label]["comp"] >= comp_th and cluster_stats[label]["cont"] < cont_th:
            hq += 1
            positive_clusters.append(label)
    return hq, positive_clusters



def run_kmedoids(X):
    import kmedoids
    breakpoint()
    #D = tf.math.sum((X[:,None]-X[None])**2, axis=-1)
    D = 0.5 - tf.math.multiply(X, X)

def compute_clusters_and_stats(
    X,
    node_names,
    reference_markers,
    contig_genes,
    node_to_gt_idx_label,
    gt_idx_label_to_node,
    k=0,
    clustering="vamb",
    cuda=False,
):
    from vamb.cluster import cluster as vamb_cluster
    if clustering == "vamb":
        best_cluster_to_contig = {
            i: c for (i, (n, c)) in enumerate(vamb_cluster(X.astype(np.float32), node_names, cuda=cuda))
        }
        best_contig_to_bin = {}
        for b in best_cluster_to_contig:
            for contig in best_cluster_to_contig[b]:
                best_contig_to_bin[contig] = b
        labels = np.array([best_contig_to_bin[n] for n in node_names])
    elif clustering == "kmedoids":
        import kmedoids
        breakpoint()
        # TODO do this on gpu if avail
        D = np.sum((X[:,None]-X[None])**2, axis=-1)
        # TODO find best k
        km = kmedoids.KMedoids(20, method='fasterpam')
        cluster_labels = km.fit_predict(D).astype(np.int64)
    elif clustering == "kmeans":
        clf = KMeans(k, random_state=1234)
        labels = clf.fit_predict(X)
        best_contig_to_bin = {node_names[i]: labels[i] for i in range(len(node_names))}
        best_cluster_to_contig = {i: [] for i in range(k)}
        for i in range(len(node_names)):
            best_cluster_to_contig[labels[i]].append(node_names[i])
    if contig_genes is not None:
        hq, positive_clusters = compute_hq(
            reference_markers=reference_markers, contig_genes=contig_genes, node_names=node_names, node_labels=labels
        )
        mq, _ = compute_hq(
            reference_markers=reference_markers,
            contig_genes=contig_genes,
            node_names=node_names,
            node_labels=labels,
            comp_th=50,
            cont_th=10,
        )
        non_comp, _ = compute_hq(
            reference_markers=reference_markers,
            contig_genes=contig_genes,
            node_names=node_names,
            node_labels=labels,
            comp_th=0,
            cont_th=10,
        )
        all_cont, _ = compute_hq(
            reference_markers=reference_markers,
            contig_genes=contig_genes,
            node_names=node_names,
            node_labels=labels,
            comp_th=90,
            cont_th=1000,
        )
        # print(hq, mq, "incompete but non cont:", non_comp, "cont but complete:", all_cont)
        positive_pairs = []
        node_names_to_idx = {node_name: i for i, node_name in enumerate(node_names)}
        for label in positive_clusters:
            for (p1, p2) in itertools.combinations(best_cluster_to_contig[label], 2):
                positive_pairs.append((node_names_to_idx[p1], node_names_to_idx[p2]))
        # print("found {} positive pairs".format(len(positive_pairs)))
        positive_pairs = np.unique(np.array(list(positive_pairs)), axis=0)
    else:
        positive_pairs, positive_clusters = None, None
        # TODO use p/r/ to get positive_clusters
        hq, mq = 0, 0
    if node_to_gt_idx_label is not None:
        p, r, f1, ari = calculate_overall_prf(
            best_cluster_to_contig, best_contig_to_bin, node_to_gt_idx_label, gt_idx_label_to_node
        )
    else:
        p, r, f1, ari = 0, 0, 0, 0

    return (
        labels,
        {"precision": p, "recall": r, "f1": f1, "ari": ari, "hq": hq, "mq": mq, "n_clusters": len(np.unique(labels))},
        positive_pairs,
        positive_clusters,
    )


def filter_disconnected(adj, node_names, markers):
    # get idx of nodes that are connected or have at least one marker
    graph = nx.convert_matrix.from_scipy_sparse_matrix(adj, edge_attribute="weight")
    # breakpoint()
    nodes_to_remove = set()
    for n1 in graph.nodes:
        if len(list(graph.neighbors(n1))) == 0 and (
            node_names[n1] not in markers or len(markers[node_names[n1]]) == 0
        ):
            nodes_to_remove.add(n1)
    graph.remove_nodes_from(list(nodes_to_remove))
    print(len(nodes_to_remove), "out of", len(node_names), "nodes without edges and markers")
    return set(graph.nodes())



def prepare_data_for_vae(dataset):
    # less preparation necessary than for GNN
    node_raw = np.hstack((dataset.node_depths, dataset.node_kmers))
    ab_dim = dataset.node_depths.shape[1]
    kmer_dim = dataset.node_kmers.shape[1]
    X = node_raw
    return X, ab_dim, kmer_dim

def prepare_data_for_gnn(
    dataset, use_edge_weights=True, use_disconnected=True, cluster_markers_only=False, use_raw=False
):
    if use_raw:
        node_raw = np.hstack((dataset.node_depths, dataset.node_kmers))
        #node_raw = (node_raw - node_raw.mean(axis=0, keepdims=True)) / node_raw.std(axis=0, keepdims=True)
        ab_dim = dataset.node_depths.shape[1]
        kmer_dim = dataset.node_kmers.shape[1]
        X = node_raw
    else:
        node_features = (dataset.node_embs - dataset.node_embs.mean(axis=0, keepdims=True)) / dataset.node_embs.std(
            axis=0, keepdims=True
        )
        X = node_features
        ab_dim, kmer_dim = 0, 0

    depth = 2
    # adjacency_matrix_sparse, edge_features = filter_graph_with_markers(adjacency_matrix_sparse, node_names, contig_genes, edge_features, depth=depth) 
    if cluster_markers_only and dataset.contig_markers is not None:
        connected_marker_nodes = filter_disconnected(dataset.adj_matrix, dataset.node_names, dataset.contig_markers)
        nodes_with_markers = [
            i
            for i, n in enumerate(dataset.node_names)
            if n in dataset.contig_markers and len(dataset.contig_markers[n]) > 0
        ]
        # print("eval with ", len(nodes_with_markers), "contigds")
        cluster_mask = [n in nodes_with_markers for n in range(len(dataset.node_names))]
    else:
        cluster_mask = [True] * len(dataset.node_names)
        connected_marker_nodes = set(range(len(dataset.node_names)))
    adj_norm = normalize_adj_sparse(dataset.adj_matrix)
    if use_edge_weights:
        edge_features = (dataset.edge_weights - dataset.edge_weights.min()) / (
            dataset.edge_weights.max() - dataset.edge_weights.min()
        )
        old_rows, old_cols = dataset.adj_matrix.row, dataset.adj_matrix.col
        old_idx_to_edge_idx = {(r, c): i for i, (r, c) in enumerate(zip(old_rows, old_cols))}
        old_values = adj_norm.data.astype(np.float32)
        new_values = []
        for i, j, ov in zip(adj_norm.row, adj_norm.col, old_values):
            if i == j:
                new_values.append(1.0)
            else:
                try:
                    eidx = old_idx_to_edge_idx[(i, j)]
                    new_values.append(ov * edge_features[eidx])
                except:
                    new_values.append(ov)
        new_values = np.array(new_values).astype(np.float32)
    else:
        new_values = adj_norm.data.astype(np.float32)
    adj = tf.SparseTensor(
        indices=np.array([adj_norm.row, adj_norm.col]).T, values=new_values, dense_shape=adj_norm.shape
    )
    adj = tf.sparse.reorder(adj)
    if not use_disconnected:
        train_edges = [
            eid
            for eid in range(len(adj_norm.row))
            if adj_norm.row[eid] in connected_marker_nodes and adj_norm.col[eid] in connected_marker_nodes
        ]

        train_adj = tf.SparseTensor(
            indices=np.array([adj_norm.row[train_edges], adj_norm.col[train_edges]]).T,
            values=new_values[train_edges],
            dense_shape=adj_norm.shape,
        )
        train_adj = tf.sparse.reorder(train_adj)

    else:
        train_adj = adj
    # neg_pair_idx = None
    pos_pair_idx = None
    print("**** Num of edges:", train_adj.indices.shape[0])
    return X, adj, train_adj, cluster_mask, dataset.neg_pairs_idx, pos_pair_idx, ab_dim, kmer_dim

def save_model(args, epoch, th, th_vae):
    if th_vae is not None:
        # save encoder and decoder
        th_vae.encoder.save(os.path.join(args.outdir, args.outname + "_best_encoder"))
        th_vae.decoder.save(os.path.join(args.outdir, args.outname + "_best_decoder"))
    if th is not None:
        th.gnn_model.save(os.path.join(args.outdir, args.outname + "_best_gnn"))


    
class TensorboardLogger(logging.StreamHandler):
    def __init__(self, file_writer, runname=""):
        logging.StreamHandler.__init__(self)
        self.file_writer = file_writer
        self.runname = runname
        self.step = 0

    def emit(self, msg):
        with self.file_writer.as_default():
            tf.summary.text(self.runname, msg.msg, step=self.step)
        self.step += 1

def run_model(dataset, args, logger):
    set_seed(args.seed)
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
    gname = args.model_name
    if gname == "vae":
        args.ae_only = True
    else:
        gmodel_type = name_to_model[gname.split("_")[0].upper()]
    clustering = args.clusteringalgo
    k = args.kclusters
    use_edge_weights = True
    use_disconnected = False
    cluster_markers_only = True
    decay = 0.5 ** (2.0 / epochs)
    concat_features = args.concat_features
    use_ae = gname.endswith("_ae") or args.ae_only or gname == "vae"

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(args.outdir, 'logs/' + args.outname + current_time + '/train')
    summary_writer = tf.summary.create_file_writer(train_log_dir)
    print("logging to tensorboard")
    tb_handler = TensorboardLogger(summary_writer, runname=args.outname + current_time)
    logger.addHandler(tb_handler)
    #tf.summary.trace_on(graph=True)

    logger.info("******* Running model: {} **********".format(gname))
    logger.info("***** using edge weights: {} ******".format(use_edge_weights))
    logger.info("***** using disconnected: {} ******".format(use_disconnected))
    logger.info("***** concat features: {} *****".format(concat_features))
    logger.info("***** cluster markers only: {} *****".format(cluster_markers_only))

    if use_ae:
        args.rawfeatures = True
    logger.info("***** Using raw kmer+abund features: {}".format(args.rawfeatures))
    tf.config.experimental_run_functions_eagerly(True)

    if not args.ae_only:
        X, adj, train_adj, cluster_mask, neg_pair_idx, pos_pair_idx, ab_dim, kmer_dim = prepare_data_for_gnn(
            dataset, use_edge_weights, use_disconnected, cluster_markers_only, use_raw=args.rawfeatures
        )
        logger.info("***** SCG neg pairs: {}".format(neg_pair_idx.shape))
    else:
        X, ab_dim, kmer_dim = prepare_data_for_vae(dataset)
        cluster_mask = [True] * len(dataset.node_names)

    logger.info("***** input features dimension: {}".format(X[cluster_mask].shape))
    

    # pre train clustering
    if not args.skip_preclustering:
        cluster_labels, stats, _, hq_bins = compute_clusters_and_stats(
                    X[cluster_mask],
                    node_names[cluster_mask],
                    dataset.ref_marker_sets,
                    dataset.contig_markers,
                    dataset.node_to_label,
                    dataset.label_to_node,
                    clustering=clustering,
                    k=k,
                    #cuda=args.cuda,
                )
        if args.tsne:
            cluster_to_contig = {cluster: [dataset.node_names[i] for i,x in enumerate(cluster_labels) if x == cluster] for cluster in set(cluster_labels)}
            node_embeddings_2dim, centroids_2dim = run_tsne(X, dataset, cluster_to_contig, hq_bins, centroids=None)
            plot_embs(
                dataset.node_names,
                node_embeddings_2dim,
                dataset.label_to_node.copy(),
                centroids=centroids_2dim,
                hq_centroids=hq_bins,
                node_sizes=None,
                outputname=os.path.join(args.outdir, f"{args.outname}_tsne_clusters_notrain.png"),
            )
        logger.info(f">>> Pre train stats: {str(stats)}")
    
    pname = ""

    scores = []
    losses = {"total": [], "ae": [], "gnn": [], "scg": []}
    all_cluster_labels = []
    X = X.astype(np.float32)
    features = tf.constant(X)
    if not use_ae:
        input_dim_gnn = X.shape[1]
    else:
        input_dim_gnn = output_dim_vae
    logger.info(f"*** Model input dim {X.shape[1]}, GNN input dim {input_dim_gnn}, use_ae: {use_ae}, run AE only: {args.ae_only}")
    
    S = []
    if not args.ae_only:
        gnn_model = gmodel_type(
            features_shape=features.shape,
            input_dim=input_dim_gnn,
            labels=None,
            adj=train_adj,
            n_labels=output_dim_gnn,
            hidden_units=hidden_gnn,
            layers=nlayers_gnn,
            conv_last=False,
        )  # , use_bn=True, use_vae=False)
        logger.info(f"*** output clustering dim {output_dim_gnn}")
    else:
        gnn_model = None
        logger.info(f"*** output clustering dim {output_dim_vae}")
    
    if use_ae:
        encoder = VAEEncoder(ab_dim, kmer_dim, hidden_vae, zdim=output_dim_vae, dropout=args.dropout_vae)
        decoder = VAEDecoder(ab_dim, kmer_dim, hidden_vae, zdim=output_dim_vae, dropout=args.dropout_vae)
        th_vae = TrainHelperVAE(encoder, decoder, learning_rate=lr_vae, kld_weight=1/args.kld_alpha)
    else:
        encoder = None
        decoder = None
        th_vae = None

    if not args.ae_only:
        th = TH(
            features,
            gnn_model=gnn_model,
            lr=lr_gnn,
            all_different_idx=neg_pair_idx,
            all_same_idx=pos_pair_idx,
            ae_encoder=encoder,
            ae_decoder=decoder,
            latentdim=output_dim_gnn,
            gnn_weight=float(args.gnn_alpha),
            ae_weight=float(args.ae_alpha),
            scg_weight=float(args.scg_alpha),
            num_negatives=args.negatives,
            decoder_input=args.decoder_input,
        )
    else:
        th = None

    if not args.quiet:
        if not args.ae_only:
            gnn_model.summary()
        #if gname.endswith("_ae"):
        #    th.encoder.summary()
        #    th.decoder.summary()
    if args.eval_split == 0:
        train_idx = np.arange(len(features))
        eval_idx = []
    else:
        train_idx = np.array(random.sample(list(range(len(features))), int(len(features)*(1-args.eval_split))))
        eval_idx = np.array([x for x in np.arange(len(features)) if x not in train_idx])
        logging.info(f"**** using {len(train_idx)} for training and {len(eval_idx)} for eval")
    features = np.array(features)
    pbar_epoch = tqdm(range(epochs), disable=args.quiet, position=0)
    scores = []
    best_embs = None
    best_model = None
    best_hq = 0
    best_epoch = 0
    batch_size = args.batchsize
    if batch_size == 0:
        batch_size = len(train_idx)
    if use_ae:
        logger.info("**** initial batch size: {} ****".format(batch_size))
        batch_steps = [25, 75, 150, 300]
        batch_steps = [x for i, x in enumerate(batch_steps) if (2 ** (i+1))*batch_size < len(train_idx)]
        logger.info("**** epoch batch size doubles: {} ****".format(str(batch_steps)))
    vae_losses = []
    step = 0
    for e in pbar_epoch:
        vae_epoch_losses = {"kld": [], "total": [], "kmer": [], "abundance": []}
        np.random.shuffle(train_idx)
        recon_loss = 0
        if use_ae:
            # train VAE in batches
            if e in batch_steps:
                #print(f'Increasing batch size from {batch_size:d} to {batch_size*2:d}')
                batch_size = batch_size * 2
            np.random.shuffle(train_idx)
            n_batches = len(train_idx)//batch_size
            pbar_vaebatch = tqdm(range(n_batches), disable=(args.quiet or batch_size == len(train_idx) or n_batches < 100), position=1, ascii=' =')
            for b in pbar_vaebatch:
                batch_idx = train_idx[b*batch_size:(b+1)*batch_size]
                vae_losses = th_vae.train_step(X[batch_idx], summary_writer, step, vae=True)
                vae_epoch_losses["total"].append(vae_losses[0])
                vae_epoch_losses["kmer"].append(vae_losses[1])
                vae_epoch_losses["abundance"].append(vae_losses[2])
                vae_epoch_losses["kld"].append(vae_losses[3])
                pbar_vaebatch.set_description(f'E={e} L={np.mean(vae_epoch_losses["total"][-10:]):.4f}')
                step += 1
            with summary_writer.as_default():
                tf.summary.scalar('train loss', np.mean(vae_epoch_losses["total"]), step=step)
                tf.summary.scalar('train kmer loss', np.mean(vae_epoch_losses["kmer"]), step=step)
                tf.summary.scalar('train ab loss', np.mean(vae_epoch_losses["abundance"]), step=step)
                tf.summary.scalar('train kld loss', np.mean(vae_epoch_losses["kld"]), step=step)
            if args.eval_split > 0:
                eval_mu, eval_logsigma = th_vae.encoder(X[eval_idx], training=False)
                eval_mse1, eval_mse2, eval_kld = th_vae.loss(X[eval_idx], eval_mu, eval_logsigma, vae=True, training=False)
                eval_loss = eval_mse1 + eval_mse2 - eval_kld
                with summary_writer.as_default():
                    tf.summary.scalar('eval loss', eval_loss, step=step)
                    tf.summary.scalar('eval kmer loss', eval_mse2, step=step)
                    tf.summary.scalar('eval ab loss', eval_mse1, step=step)
                    tf.summary.scalar('eval kld loss', eval_kld, step=step)
            else:
                eval_loss, eval_mse1, eval_mse2, eval_kld = 0, 0, 0, 0
            recon_loss = np.mean(vae_epoch_losses["total"])
        else:
            step += 1
            
        with summary_writer.as_default():
            tf.summary.scalar('epoch', e, step=step)
        if not args.ae_only:
            total_loss, gnn_loss, diff_loss = th.train_unsupervised(train_idx)
            with summary_writer.as_default():
                tf.summary.scalar('gnn loss', gnn_loss, step=step)
                tf.summary.scalar('SCG loss', diff_loss, step=step)
                tf.summary.scalar('Total loss', total_loss, step=step)
            gnn_loss = gnn_loss.numpy()
            diff_loss = diff_loss.numpy()
        else:
            gnn_loss = 0
            diff_loss = 0
        #if 
        #gpu_mem_alloc = tf.config.experimental.get_memory_info('GPU:0')["peak"] / 1000000 if args.cuda else 0
        gpu_mem_alloc = tf.config.experimental.get_memory_usage('GPU:0') / 1000000 if args.cuda else 0
        if (e + 1) % RESULT_EVERY == 0:
            if not args.ae_only:
                th.gnn_model.adj = adj
            eval_features = features
            if use_ae:
                eval_features = encoder(features)[0]
            if not args.ae_only:
                node_new_features = th.gnn_model(eval_features, None, training=False)
                node_new_features = node_new_features.numpy()
            else:
                node_new_features = eval_features.numpy()

            with summary_writer.as_default():
                tf.summary.scalar('Embs average', np.mean(node_new_features), step=step)
                tf.summary.scalar('Embs std', np.std(node_new_features), step=step)
            # concat with original features
            if concat_features:
                node_new_features = tf.concat([features, node_new_features], axis=1).numpy()

            cluster_labels, stats, _, hq_bins = compute_clusters_and_stats(
                node_new_features[cluster_mask],
                node_names[cluster_mask],
                dataset.ref_marker_sets,
                dataset.contig_markers,
                dataset.node_to_label,
                dataset.label_to_node,
                clustering=clustering,
                k=k,
                #cuda=args.cuda,
            )
            #plot tSNE
            if args.tsne:
                cluster_to_contig = {cluster: [dataset.node_names[i] for i,x in enumerate(cluster_labels) if x == cluster] for cluster in set(cluster_labels)}
                node_embeddings_2dim, centroids_2dim = run_tsne(node_new_features, dataset, cluster_to_contig, hq_bins, centroids=None)
                plot_embs(
                    dataset.node_names,
                    node_embeddings_2dim,
                    dataset.label_to_node.copy(),
                    centroids=centroids_2dim,
                    hq_centroids=hq_bins,
                    node_sizes=None,
                    outputname=os.path.join(args.outdir, f"{args.outname}_tsne_clusters_epoch_{e}.png"),
                )
            
            stats["epoch"] = e
            scores.append(stats)
            logger.info(str(stats))
            with summary_writer.as_default():
                tf.summary.scalar('hq_bins',  stats["hq"], step=step)
            all_cluster_labels.append(cluster_labels)
            if not args.ae_only:
                th.gnn_model.adj = train_adj

            if dataset.contig_markers is not None and stats["hq"] > best_hq:
                best_hq = stats["hq"]
                #best_model = th.gnn_model
                best_embs = node_new_features
                best_epoch = e
                #save_model(args, e, th, th_vae)

            elif dataset.contig_markers is None and stats["f1"] > best_hq:
                best_hq = stats["f1"]
                #best_model = th.gnn_model
                best_embs = node_new_features
                best_epoch = e
                #save_model(args, e, th, th_vae)
            # print('--- END ---')
            if args.quiet:
                logger.info(f"--- EPOCH {e:d} ---")
                logger.info(f"[{gname} {nlayers_gnn}l {pname}] L={gnn_loss:.3f} D={diff_loss:.3f} R={recon_loss:.3f} BestHQ={best_hq} Best Epoch={best_epoch} Max GPU MB={gpu_mem_alloc:.1f}")
                logger.info(str(stats))

        pbar_epoch.set_description(
            f"[{gname} {nlayers_gnn}l {pname}] L={gnn_loss:.3f} D={diff_loss:.3f} R={recon_loss:.3f} BestHQ={best_hq} Best Epoch={best_epoch} Max GPU MB={gpu_mem_alloc:.1f}"
        )
        total_loss = gnn_loss + diff_loss + recon_loss
        losses["gnn"].append(gnn_loss)
        losses["scg"].append(diff_loss)
        losses["ae"].append(recon_loss)
        losses["total"].append(total_loss)
        #if e == 1:
            #breakpoint()
        #    with summary_writer.as_default():
        #        tf.summary.trace_export(args.outname, step=0, profiler_outdir=train_log_dir) 
        #        summary_writer.flush()

    """if use_ae:
        eval_features = encoder(features)[0]
    if not args.ae_only:
        th.gnn_model.adj = adj
        node_new_features = th.gnn_model(eval_features, None, training=False)
        node_new_features = node_new_features.numpy()
    else:
        node_new_features = eval_features.numpy()

    # concat with original features
    if concat_features:
        node_new_features = tf.concat([features, node_new_features], axis=1).numpy()"""
    if best_embs is None:
        best_embs = node_new_features
    
    cluster_labels, stats, _, _ = compute_clusters_and_stats(
        best_embs[cluster_mask],
        node_names[cluster_mask],
        dataset.ref_marker_sets,
        dataset.contig_markers,
        dataset.node_to_label,
        dataset.label_to_node,
        clustering=clustering,
        k=k,
        #cuda=args.cuda,
    )
    stats["epoch"] = e
    scores.append(stats)
    # get best stats:
    # if concat_features:  # use HQ
    hqs = [s["hq"] for s in scores]
    epoch_hqs = [s["epoch"] for s in scores]
    best_idx = np.argmax(hqs)
    # else:  # use F1
    #    f1s = [s["f1"] for s in scores]
    #    best_idx = np.argmax(f1s)
    # S.append(stats)
    S.append(scores[best_idx])
    logger.info(f">>> best epoch: {RESULT_EVERY + (best_idx*RESULT_EVERY)} : {scores[best_idx]} <<<")
    with open(f"{dataset.name}_{gname}_{clustering}{k}_{nlayers_gnn}l_{pname}_results.tsv", "w") as f:
        f.write("@Version:0.9.0\n@SampleID:SAMPLEID\n@@SEQUENCEID\tBINID\n")
        for i in range(len(cluster_labels)):
            f.write(f"{node_names[i]}\t{cluster_labels[i]}\n")
    #del gnn_model, th
    # res_table.add_row(f"{gname} {clustering}{k} {nlayers}l {pname}", S)
    # if gt_idx_label_to_node is not None:
    #    # save embs
    #    np.save(f"{dataset}_{gname}_{clustering}{k}_{nlayers}l_{pname}_embs.npy", node_new_features)
    # plot_clusters(node_new_features, features.numpy(), cluster_labels, f"{gname} VAMB {nlayers}l {pname}")
    # res_table.show()
    #breakpoint()
    #plt.plot(range(len(losses["total"])), losses["total"], label="total loss")
    if not args.quiet:
        fig, ax1 = plt.subplots()
        ax1.plot(range(1, len(losses["gnn"])), losses["gnn"][1:], label="GNN loss")
        ax1.plot(range(1, len(losses["scg"])), losses["scg"][1:], label="SCG loss")
        ax1.plot(range(1, len(losses["ae"])), losses["ae"][1:], label="AE loss")
        ax1.legend(loc='upper right')
        ax1.set_ylim(0,1.5)
        ax2 = ax1.twinx()
        #ax2.plot(epoch_hqs, [hq/max(hqs) for hq in hqs], label="HQ", color='red', marker='o')
        ax2.plot(epoch_hqs, hqs, label="HQ", color='red', marker='o')
        plt.xlabel("epoch")
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(args.outdir,args.outname + "_training.png"), dpi=500)
        plt.show()
        logger.info("saving figure to {}".format(os.path.join(args.outdir, args.outname + "_training.png")))
        
    return best_embs, scores[best_idx]
