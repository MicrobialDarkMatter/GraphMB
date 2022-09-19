from collections import Counter
from sklearn.cluster import KMeans, MiniBatchKMeans
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
from scipy.sparse import coo_matrix, diags

from graphmb.models import SAGE, SAGELAF, GCN, GCNLAF, GAT, GATLAF, TH, TrainHelperVAE, VAEDecoder, VAEEncoder, VGAE
from graph_functions import set_seed, run_tsne, plot_embs, plot_edges_sim
from graphmb.evaluate import calculate_overall_prf

name_to_model = {"SAGE": SAGE, "SAGELAF": SAGELAF, "GCN": GCN, "GCNLAF": GCNLAF, "GAT": GAT, "GATLAF": GATLAF}


def normalize_adj_sparse(A):
    #breakpoint()
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

def compute_unresolved(reference_markers, contig_genes, node_names, node_labels, resolved_clusters):
    # completeness of a bin with all not HQ -> how many more bins we could get
    unresolved_contigs = np.array([n for i,n in enumerate(node_names) if node_labels[i] not in resolved_clusters])
    cluster_stats = compute_cluster_score(reference_markers, contig_genes,
                                          unresolved_contigs, np.ones(len(unresolved_contigs)))
    contamination = cluster_stats[1]["cont"]
    potential_mags = int(contamination / 100)
    return potential_mags

def run_kmedoids(X):
    import kmedoids
    breakpoint()
    #D = tf.math.sum((X[:,None]-X[None])**2, axis=-1)
    D = 0.5 - tf.math.multiply(X, X)

def compute_clusters_and_stats(
    X,
    node_names,
    dataset,
    k=0,
    clustering="vamb",
    cuda=False,
    tsne=False,
    tsne_path=None,
    max_pos_pairs=None,
    use_labels=False
):
    reference_markers = dataset.ref_marker_sets
    contig_genes = dataset.contig_markers
    node_to_gt_idx_label = dataset.node_to_label
    gt_idx_label_to_node = dataset.label_to_node

    from vamb.cluster import cluster as vamb_cluster
    if clustering == "vamb":
        #breakpoint()
        cluster_to_contig = {
            i: c for (i, (n, c)) in enumerate(vamb_cluster(X.astype(np.float32), node_names, cuda=cuda))
        }
        contig_to_bin = {}
        for b in cluster_to_contig:
            for contig in cluster_to_contig[b]:
                contig_to_bin[contig] = b
        labels = np.array([contig_to_bin[n] for n in node_names])
        cluster_to_embs = {
            c: np.array([X[i] for i, n in enumerate(node_names) if n in cluster_to_contig[c]])
            for c in cluster_to_contig
        }
        cluster_centroids = np.array([cluster_to_embs[c].mean(0) for c in cluster_to_contig])
    elif clustering == "kmeansbatch":
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=2048, verbose=0) #, init=seed_matrix)
        labels = kmeans.fit_predict(X)
        contig_to_bin = {node_names[i]: labels[i] for i in range(len(node_names))}
        cluster_to_contig = {i: [] for i in range(k)}
        for i in range(len(node_names)):
            cluster_to_contig[labels[i]].append(node_names[i])
        #cluster_centroids = kmeans.cluster_centers_
    elif clustering == "kmeansgpu":
        pass
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
        contig_to_bin = {node_names[i]: labels[i] for i in range(len(node_names))}
        cluster_to_contig = {i: [] for i in range(k)}
        for i in range(len(node_names)):
            cluster_to_contig[labels[i]].append(node_names[i])
    if contig_genes is not None and len(contig_genes) > 0:
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
        unresolved_mags = compute_unresolved(reference_markers=reference_markers,
            contig_genes=contig_genes,
            node_names=node_names,
            node_labels=labels,
            resolved_clusters=positive_clusters)
        # print(hq, mq, "incompete but non cont:", non_comp, "cont but complete:", all_cont)
        positive_pairs = get_positive_pairs(node_names, positive_clusters, cluster_to_contig, max_pos_pairs)

    else:
        positive_pairs, positive_clusters, unresolved_mags = None, None, 0
        # TODO use p/r/ to get positive_clusters
        hq, mq = 0, 0
    if node_to_gt_idx_label is not None:
        p, r, f1, ari = calculate_overall_prf(
            cluster_to_contig, contig_to_bin, node_to_gt_idx_label, gt_idx_label_to_node
        )
    else:
        p, r, f1, ari = 0, 0, 0, 0

    #plot tSNE
    if tsne:
        cluster_to_contig = {cluster: [dataset.node_names[i] for i,x in enumerate(labels) if x == cluster] for cluster in set(labels)}
        node_embeddings_2dim, centroids_2dim = run_tsne(X, dataset, cluster_to_contig, positive_clusters, centroids=cluster_centroids)
        plot_embs(
            dataset.node_names,
            node_embeddings_2dim,
            #dataset.label_to_node.copy(),
            cluster_to_contig,
            centroids=centroids_2dim,
            hq_centroids=positive_clusters,
            node_sizes=None,
            outputname=tsne_path,
        )
    return (
        labels,
        {"precision": p, "recall": r, "f1": f1, "ari": ari, "hq": hq, "mq": mq,
        "n_clusters": len(np.unique(labels)), "unresolved": unresolved_mags},
        positive_pairs,
        positive_clusters,
    )

def get_positive_pairs(node_names, positive_clusters, cluster_to_contig, max_pos_pairs=None):
    positive_pairs = []
    node_names_to_idx = {node_name: i for i, node_name in enumerate(node_names)}
    for label in positive_clusters:
        added_pairs = []
        for (p1, p2) in itertools.combinations(cluster_to_contig[label], 2):
            added_pairs.append((node_names_to_idx[p1], node_names_to_idx[p2]))    
        if max_pos_pairs is not None and len(added_pairs) > max_pos_pairs:
            added_pairs = random.sample(added_pairs, max_pos_pairs)
        positive_pairs.extend(added_pairs)
    # print("found {} positive pairs".format(len(positive_pairs)))
    positive_pairs = np.unique(np.array(list(positive_pairs)), axis=0)
    return positive_pairs


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


def prepare_data_for_gnn(
    dataset, use_edge_weights=True, cluster_markers_only=False, use_raw=False,
    binarize=False, remove_edges=False, remove_same_scg=True
):
    if use_raw: # use raw features instead of precomputed embeddings
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
        #connected_marker_nodes = set(range(len(dataset.node_names)))
    
    adj_matrix = dataset.adj_matrix.copy()
    edge_weights = dataset.edge_weights.copy()
    if binarize:
        # both dataset.adj_matrix and dataset.edge_weights
        #breakpoint()
        percentile = 90
        threshold = np.percentile(dataset.edge_weights, percentile)
        print(f"using this threshold ({percentile} percentile) {threshold} on adj matrix with {len(dataset.adj_matrix.row)} edges")
        #dataset.adj_matrix = dataset.adj_matrix
        adj_matrix.data[adj_matrix.data < threshold] = 0
        adj_matrix.data[adj_matrix.data >= threshold] = 1
        adj_matrix.eliminate_zeros()
        edge_weights = adj_matrix.data 
        #dataset.edge_weights = np.ones(len(dataset.adj_matrix.row))
        print(f"reduce matrix to {len(edge_weights)} edges")
    
    if remove_same_scg:
        edges_with_same_scgs = 0
        scg_counter = Counter()
        for x, (i, j) in enumerate(zip(adj_matrix.row, adj_matrix.col)):
            overlap = len(dataset.contig_markers[dataset.node_names[i]].keys() & \
                dataset.contig_markers[dataset.node_names[j]].keys())
            if overlap > 0 and i != j:
                #remove edge
                scg_counter[overlap] += 1
                adj_matrix.data[x] = 0
                edge_weights[x] = 0
                edges_with_same_scgs += 1
        adj_matrix.eliminate_zeros()
        print(f"deleted {edges_with_same_scgs} edges with same SCGs")
        print(scg_counter.most_common(20))
    if remove_edges:
        # create self loops only sparse adj matrix
        n = len(dataset.node_names)
        adj_matrix = coo_matrix((np.ones(n), (np.array(range(n)), np.array(range(n)))), shape=(n,n))
        edge_weights = np.ones(len(adj_matrix.row))
        print(f"reduce matrix to {len(edge_weights)} edges")
    adj_norm = normalize_adj_sparse(adj_matrix)
    if use_edge_weights:
        #edge_features = (dataset.edge_weights - dataset.edge_weights.min()) / (
        #    dataset.edge_weights.max() - dataset.edge_weights.min()
        #)
        edge_features = edge_weights / edge_weights.max()
        # multiply normalized values by edge weights
        old_rows, old_cols = adj_matrix.row, adj_matrix.col
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
        adj_norm.data = np.ones(len(adj_norm.row))
        #new_values = adj_norm.data.astype(np.float32)
    adj = tf.SparseTensor(
        indices=np.array([adj_norm.row, adj_norm.col]).T, values=new_values, dense_shape=adj_norm.shape
    )
    adj = tf.sparse.reorder(adj)

    # neg_pair_idx = None
    pos_pair_idx = None
    print("**** Num of edges:", adj.indices.shape[0])
    return X, adj, cluster_mask, dataset.neg_pairs_idx, pos_pair_idx, ab_dim, kmer_dim

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


def log_to_tensorboard(writer, values, step):
    """Write key-values to writer
    """
    for k, v in values.items():
        with writer.as_default():
            tf.summary.scalar(k, v, step=step)


def eval_epoch(logger, summary_writer, node_new_features, cluster_mask, weights,
               step, args, dataset, epoch, scores, best_hq, best_embs, best_epoch, best_model):
    log_to_tensorboard(summary_writer, {"Embs average": np.mean(node_new_features), 'Embs std': np.std(node_new_features) }, step)

    cluster_labels, stats, _, hq_bins = compute_clusters_and_stats(
        node_new_features[cluster_mask], np.array(dataset.node_names)[cluster_mask],
        dataset, clustering=args.clusteringalgo, k=args.kclusters, tsne=args.tsne, #cuda=args.cuda,
    )
    
    stats["epoch"] = epoch
    scores.append(stats)
    #logger.info(str(stats))

    log_to_tensorboard(summary_writer, {"hq_bins": stats["hq"], "mq_bins": stats["mq"]}, step)
    #all_cluster_labels.append(cluster_labels)

    if dataset.contig_markers is not None and stats["hq"] > best_hq:
        best_hq, best_embs, best_epoch, best_model = stats["hq"], node_new_features, epoch, weights
        #best_model = th.gnn_model
        #save_model(args, e, th, th_vae)

    elif dataset.contig_markers is None and stats["f1"] > best_hq:
        best_hq, best_embs, best_epoch, best_model = stats["f1"], node_new_features, epoch, weights
        #best_model = th.gnn_model
        #save_model(args, e, th, th_vae)
    # print('--- END ---')
    #if args.quiet:
    #    logger.info(f"--- EPOCH {e:d} ---")
    #    logger.info(f"[{gname} {nlayers_gnn}l] L={gnn_loss:.3f} D={diff_loss:.3f} HQ={stats['hq']} BestHQ={best_hq} Best Epoch={best_epoch} Max GPU MB={gpu_mem_alloc:.1f}")
    #    logger.info(str(stats))

    #cluster_labels, stats, _, hq_bins = compute_clusters_and_stats(
    #    node_new_features, np.array(dataset.node_names),
    #    dataset, clustering=args.clusteringalgo, k=args.kclusters, tsne=args.tsne, #cuda=args.cuda,
    #)
    #log_to_tensorboard(summary_writer, {"hq_bins_all": stats["hq"], "mq_bins_all": stats["mq"]}, step)

    return best_hq, best_embs, best_epoch, scores, best_model


def eval_epoch_cluster(logger, summary_writer, node_new_features, cluster_mask, best_hq,
               step, args, dataset, epoch, tsne, cluster=True):
    
    if cluster:
        log_to_tensorboard(summary_writer, {"Embs average": np.mean(node_new_features), 'Embs std': np.std(node_new_features) }, step)
        tsne_path = os.path.join(args.outdir, f"{args.outname}_tsne_clusters_epoch_{epoch}.png")
        cluster_labels, stats, positive_pairs, hq_bins = compute_clusters_and_stats(
            node_new_features[cluster_mask], np.array(dataset.node_names)[cluster_mask],
            dataset, clustering=args.clusteringalgo, k=args.kclusters, tsne=tsne, tsne_path=tsne_path, max_pos_pairs=None #cuda=args.cuda,
        )
    else:
        breakpoint()
        cluster_labels = np.argmax(node_new_features[cluster_mask], axis=1)
        hq, positive_clusters = compute_hq(reference_markers=dataset.ref_marker_sets,
                                           contig_genes=dataset.contig_markers,
                                           node_names=np.array(dataset.node_names)[cluster_mask],
                                           node_labels=cluster_labels)
        mq, _ = compute_hq(reference_markers=dataset.ref_marker_sets,
                                           contig_genes=dataset.contig_markers,
                                           node_names=np.array(dataset.node_names)[cluster_mask],
                                           node_labels=cluster_labels, comp_th=50, cont_th=10,)
        stats = {"hq": hq, "unresolved": len(positive_clusters), "mq": mq}
        
        cluster_to_contig = {i: [] for i in range(max(cluster_labels) + 1)}
        for i in range(len(dataset.node_names)):
            cluster_to_contig[cluster_labels[i]].append(dataset.node_names[i])
        positive_pairs = None
   
    stats["epoch"] = epoch
    #logger.info(str(stats))

    log_to_tensorboard(summary_writer, {"hq_bins": stats["hq"], "mq_bins": stats["mq"]}, step)
    #all_cluster_labels.append(cluster_labels)
    new_best = False
    if dataset.contig_markers is not None and stats["hq"] > best_hq:
        new_best = True
    elif dataset.contig_markers is None and stats["f1"] > best_hq:
        new_best = True
    # print('--- END ---')
    #if args.quiet:
    #    logger.info(f"--- EPOCH {e:d} ---")
    #    logger.info(f"[{gname} {nlayers_gnn}l] L={gnn_loss:.3f} D={diff_loss:.3f} HQ={stats['hq']} BestHQ={best_hq} Best Epoch={best_epoch} Max GPU MB={gpu_mem_alloc:.1f}")
    #    logger.info(str(stats))

    #cluster_labels, stats, _, hq_bins = compute_clusters_and_stats(
    #    node_new_features, np.array(dataset.node_names),
    #    dataset, clustering=args.clusteringalgo, k=args.kclusters, tsne=args.tsne, #cuda=args.cuda,
    #)
    #log_to_tensorboard(summary_writer, {"hq_bins_all": stats["hq"], "mq_bins_all": stats["mq"]}, step)

    return stats, new_best, cluster_labels, positive_pairs

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
    use_disconnected = not args.quick
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
    logger.info("***** using disconnected: {} ******".format(use_disconnected))
    logger.info("***** concat features: {} *****".format(concat_features))
    logger.info("***** cluster markers only: {} *****".format(cluster_markers_only))
    logger.info("***** threshold adj matrix: {} *****".format(args.binarize))
    logger.info("***** self edges only: {} *****".format(args.noedges))
    logger.info("***** Using raw kmer+abund features: {}".format(args.rawfeatures))
    tf.config.experimental_run_functions_eagerly(True)


    X, adj, cluster_mask, neg_pair_idx, pos_pair_idx, ab_dim, kmer_dim = prepare_data_for_gnn(
            dataset, use_edge_weights, cluster_markers_only, use_raw=True,
            binarize=args.binarize, remove_edges=args.noedges)
    logger.info("***** SCG neg pairs: {}".format(neg_pair_idx.shape))
    logger.info("***** input features dimension: {}".format(X[cluster_mask].shape))
    # pre train clustering
    if not args.skip_preclustering:
        cluster_labels, stats, _, hq_bins = compute_clusters_and_stats(
                    X[cluster_mask], node_names[cluster_mask],
                    dataset, clustering=clustering, k=k, tsne=args.tsne, #cuda=args.cuda,
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

            best_hq, best_embs, best_epoch, scores = eval_epoch(logger, summary_writer, node_new_features,
                                                                cluster_mask, e, args, dataset, e, scores,
                                                                best_hq, best_embs, best_epoch)

  
            #stats["epoch"] = e
            #scores.append(stats)
            #logger.info(str(stats))
            #with summary_writer.as_default():
            #    tf.summary.scalar('hq_bins',  stats["hq"], step=step)
            #all_cluster_labels.append(cluster_labels)
            # print('--- END ---')
            if args.quiet:
                logger.info(f"--- EPOCH {e:d} ---")
                logger.info(f"[VGAE {nlayers_gnn}l] L={loss:.3f}  HQ={stats['hq']}  BestHQ={best_hq} Best Epoch={best_epoch} Max GPU MB={gpu_mem_alloc:.1f}")
                logger.info(str(stats))


    _, embs, _, _, _ = model((X_train, A_train), training=False)
    embs = embs.numpy()


