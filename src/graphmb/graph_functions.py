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
import operator
import scipy


# import dgl
import random
#import tensorflow

from graphmb.evaluate import read_contig_genes, read_marker_gene_sets, evaluate_contig_sets, \
                            calculate_overall_prf, calculate_sim_between_same_labels_small

# import torch

SEED = 0

colors = [
    "black",
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "yellow",
    "silver",
    "maroon",
    "fuchsia",
    "lime",
    "olive",
    "yellow",
    "navy",
    "teal",
    "steelblue",
    "darkred",
    "darkgreen",
    "darkblue",
    "darkorange",
    "lightpink",
    "lightgreen",
    "lightblue",
    "crimson",
    "darkviolet",
    "tomato",
    "tan",
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


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

kernel = np.load("kernel.npz")['arr_0']
def count_kmers(seq, k, kmer_to_id, canonical_k):
    # Used in case kmers are used as input features
    # https://stackoverflow.com/q/22428020
    #breakpoint()
    kmer_letters = set(["A", "T", "C", "G"])
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1) if set(seq[i : i + k]).issubset(kmer_letters)]
    kmers = [kmer_to_id[k] for k in kmers]
    kmer_counts = Counter(kmers)
    counts = np.array([kmer_counts[k] for k in range(canonical_k)])
    counts = counts / counts.sum()
    counts += -(1/(4**k))
    counts = np.dot(counts, kernel)
    return counts


def plot_edges_sim(X, adj, scgs, outname="", max_edge_value=150, min_edge_value=2):
    """
    X: feature matrix
    adj: adjacency matrix in sparse format
    """
    # for each pair in adj, calculate sim
    x_values = []
    y_values = []
    x_same_scgs = []
    y_same_scgs = []
    plotted_edges = set()
    for x, (i, j) in enumerate(zip(adj.row, adj.col)):
        if i != j and (i,j) not in plotted_edges and (j,i) not in plotted_edges and adj.data[x] > min_edge_value:
        #y_values.append(np.dot(X[i], X[j]))
            #y_values.append(scipy.spatial.distance.cosine(X[i], X[j]))
            plotted_edges.add((i,j))
            if len(scgs[i] & scgs[j]) > 0:
                y_same_scgs.append(np.dot(X[i], X[j])/(np.linalg.norm(X[i])*np.linalg.norm(X[j])))
                x_same_scgs.append(adj.data[x])
                #TODO plot edge weight by overlap
            else:
                y_values.append(np.dot(X[i], X[j])/(np.linalg.norm(X[i])*np.linalg.norm(X[j])))
                x_values.append(adj.data[x])
    if max_edge_value is not None:
        x_values = [min(x, max_edge_value) for x in x_values]
        x_same_scgs = [min(x, max_edge_value) for x in x_same_scgs]
    #x_values = adj.values
    #y_values = []
    #for (i, j) in adj.indices:
    #    y_values.append(np.dot(X[i], X[j]
    #))
    assert len(x_values) == len(y_values)
    import matplotlib.pyplot as plt
    plt.set_loglevel("error")
    
    plt.figure(0)
    plt.scatter(
            x_values,
            y_values, label=outname, marker=".", alpha=0.5, s=1)
    plt.scatter(
            x_same_scgs,
            y_same_scgs, label=outname+"SCG", marker="o", alpha=0.5, s=3)

    plt.xlabel("edge weight capped at {}".format(max_edge_value))

    plt.ylabel("cosine sim")
    plt.legend(loc='upper right')
    plt.savefig(outname + "edges_embs.png", dpi=500)
    #plt.show()
    plt.close()

    # dist histogram
    plt.figure(1)
    counts, edges, bars =   plt.hist(y_values, bins=50)
    plt.bar_label(bars)
    plt.savefig(outname + "embs_dists_histogram.png", dpi=500)
    plt.close()




def run_tsne(embs, dataset, cluster_to_contig, hq_bins, centroids=None):
    from sklearn.manifold import TSNE

    SEED = 0
    print("running tSNE")
    # filter only good clusters
    tsne = TSNE(n_components=2, random_state=SEED)
    if len(dataset.labels) == 1:
        label_to_node = {c: cluster_to_contig[c] for c in hq_bins}
        label_to_node["mq/lq"] = []
        for c in cluster_to_contig:
            if c not in hq_bins:
                label_to_node["mq/lq"] += list(cluster_to_contig[c])
    if centroids is not None:
        all_embs = tsne.fit_transform(np.concatenate((np.array(embs), np.array(centroids)), axis=0))
        centroids_2dim = all_embs[embs.shape[0] :]
        node_embeddings_2dim = all_embs[: embs.shape[0]]
    else:
        centroids_2dim = None
        node_embeddings_2dim = tsne.fit_transform(embs)
    return node_embeddings_2dim, centroids_2dim

def draw_nx_graph(adj, node_to_label, labels_to_node, basename, contig_sizes=None, node_titles=None, cluster_info={}):
    # draw graph with pybiz library, creates an HTML file with graph
    # del labels_to_node["NA"]
    from pyvis.network import Network

    labels_to_color = {l: colors[i % len(colors)] for i, l in enumerate(labels_to_node.keys())}
    labels_to_color["NA"] = "white"
    sorted(labels_to_node, key=lambda key: len(labels_to_node[key]), reverse=True)[: len(colors)]
    # node_labels to plot
    node_labels = {
        node: {
            "label": str(node) + ":" + str(node_to_label[node]),
            "color": labels_to_color[node_to_label[node]],
        }
        for node in node_to_label
    }
    if contig_sizes is not None:
        for n in node_labels:
            node_labels[n]["size"] = int(contig_sizes[n])

    if node_titles is not None:
        for n in node_labels:
            node_labels[n]["title"] = node_titles[n]
    #breakpoint()
    graph = nx.from_scipy_sparse_matrix(adj, parallel_edges=False, create_using=None, edge_attribute='weight')
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nx.set_node_attributes(graph, node_labels)
    nodes_to_plot = [n for n in graph.nodes() if node_to_label[n] != "NA" and len(graph.edges(n)) > 0]

    net = Network(notebook=False, height="750px", width="100%")
    net.add_nodes(
        [int(n) for n in node_labels.keys() if n in nodes_to_plot],
        label=[node_labels[n]["label"] for n in node_labels  if n in nodes_to_plot],
        size=[node_labels[n].get("size", 100000)/100_000 for n in node_labels if n in nodes_to_plot],
        color=[node_labels[n]["color"] for n in node_labels if n in nodes_to_plot],
        title=[node_labels[n].get("title", f"{cluster_info.get(node_to_label[n])}") for n in node_labels if n in nodes_to_plot],
    )
    for u, v, a in graph.edges(data=True):
        if u != v:
            #if u not in net.get_nodes() or v not in net.get_nodes():
            #    breakpoint()
            weight = float(a["weight"].item())
            if weight != 1:
                net.add_edge(int(u), int(v), color="gray", title="reads weight: {}".format(weight))
            else:
                net.add_edge(int(u), int(v))

    #net.toggle_physics(False)
    net.show_buttons()
    print("saving graph to", basename)
    net.show("{}.html".format(basename))


def connected_components(graph, node_to_label, basename, min_elems=1):
    # explore connected components
    connected = [c for c in sorted(nx.connected_components(graph), key=len, reverse=True) if len(c) > min_elems]
    print("writing components to", basename + "_node_to_component.csv")
    write_components_file(connected, basename + "_node_to_component.csv")
    multiple_contigs = 0
    mixed_contigs = 0
    for group in connected:
        multiple_contigs += 1
        group_labels = [node_to_label[c] for c in group if "edge" in c]
        group_labels = set(group_labels)
        if len(group_labels) > 1:
            mixed_contigs += 1
            # print(group, group_labels)Process some integers.

    disconnected = [c for c in sorted(nx.connected_components(graph), key=len, reverse=True) if len(c) <= min_elems]
    for group in disconnected:
        for node in group:
            graph.remove_node(node)

    print("graph density:", nx.density(graph))
    print(">1", multiple_contigs)
    print("mixed groups", mixed_contigs)
    return connected, disconnected


def write_components_file(components, outpath, minsize=2):
    """Write file mapping each contig/node to a component ID (diff for each assembly)

    Args:
        components (list): List of connected components of a graph
        outpath (str): path to write file
        minsize: minimum number of elements of a component
    """
    with open(outpath, "w") as outfile:
        for ic, c in enumerate(components):
            if len(c) < minsize:
                continue
            for node in c:
                if "edge" in node:  # ignore read nodes
                    outfile.write(f"{node}\t{ic}\n")


def read_reads_mapping_sam(readspath):
    """Open all *read*.sam files in a dir and parse to dict of Read objects
    Filter by bitflag (not primary alignment) and not empty contig names
    Args:
        readspath ([type]): [description]
    """
    contigs_to_reads = {}
    reads_dict = {}
    for reads in Path(readspath).glob("**/*reads_*.sam"):
        print("reading", reads)
        speciesname = reads.stem.split("_reads")[0]
        with reads.open() as f:
            for line in tqdm(f):
                values = line.strip().split()
                if line.startswith("@SQ"):
                    continue
                elif line.startswith("@PG"):
                    continue
                else:
                    read_name = values[0]
                    if read_name not in reads_dict:
                        reads_dict[read_name] = Read(read_name, species=speciesname)
                    contig_name = values[2]
                    mapq = int(values[4])
                    bitflag = int(values[1])
                    pos = int(values[3])
                    # seq = values[9]
                    seq = None
                    #  	not primary alignment: 8,
                    if (bitflag >> 8) & 1 == 1:
                        continue
                    if contig_name == "*":
                        continue
                    # only mapq > 0
                    if mapq > 0:
                        reads_dict[read_name].mappings.add(
                            ReadMapping(read_name, bitflag, contig_name, pos, mapq, seq)
                        )
                        # if contig_name not in contigs_to_reads:
                        #    contigs_to_reads[contig_name] = set()
                        # contigs_to_reads[contig_name].add(read_name)
    return reads_dict


def augment_graph(graph, reads_dict, add_reads=False, min_mappings=0):
    """Add read nodes to graph according to read mapping file

    Args:
        graph (networkx Graph): contig assembly graph
        reads_dict (str): Dict of read objects
        add_reads: add read nodes if True, otherwise add edges between contigs
        min_mappings: min number of mappings to add edge
    Return:
        graph with added edges
    """
    edges = []
    for r in reads_dict:
        if len(reads_dict[r].mappings) > min_mappings:
            if add_reads:
                for m in reads_dict[r].mappings:
                    edges.append((r, m.contigname))
            else:
                for c1, c2 in itertools.combinations(reads_dict[r].mappings, 2):
                    if c1.contigname != c2.contigname:
                        edges.append((c1.contigname, c2.contigname, {"Title": r, "width": 3}))
    print("adding to networkx graph", len(edges), "edges")
    graph.add_edges_from(edges)
    return graph


def get_kmer_to_id(kmer, combine_revcomp=False):
    kmer_to_ids = {}
    BASE_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
    all_kmers = itertools.product("ACGT", repeat=kmer)
    all_kmers = ["".join(k) for k in all_kmers]
    new_id = 0
    for kmer in all_kmers:
        if kmer not in kmer_to_ids:
            kmer_to_ids[kmer] = new_id
            if combine_revcomp:
                rev_compl = "".join(tuple([BASE_COMPLEMENT[x] for x in reversed(kmer)]))
                kmer_to_ids[rev_compl] = new_id
            new_id += 1
    return kmer_to_ids, new_id


def open_gfa_file(filename, filter=1000, root=False, kmer=4):
    G = nx.Graph()
    if root:
        root_node = G.add_node("root", length=0)
    skipped_contigs = set()
    skipped_edges = set()
    kmer_to_id, canonical_k = get_kmer_to_id(kmer)
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("S"):
                values = line.strip().split()
                contigid = values[1]
                # contiglen = int(values[3].split(":")[-1])
                contiglen = len(values[2])
                contig_kmers = count_kmers(values[2], kmer, kmer_to_id, canonical_k)
                if contiglen < filter:
                    skipped_contigs.add(contigid)
                else:

                    G.add_node(contigid, length=contiglen, kmers=contig_kmers)  # TODO: add coverage and kmer too
                    if root:
                        G.add_edge(contigid, "root")
            if line.startswith("L"):
                values = line.strip().split()
                if values[1] in skipped_contigs or values[3] in skipped_contigs:
                    skipped_edges.add((values[1], values[3]))
                    continue
                G.add_edge(values[1], values[3])

    print("load graph from GFA")
    print("skipped these contigs {} (len<{})".format(len(skipped_contigs), filter))
    print("skipped these edges {} (len<{})".format(len(skipped_edges), filter))
    return G


def plot_embs(node_ids, node_embeddings_2dim, labels_to_node, centroids, hq_centroids, node_sizes, outputname=None):
    """Plot embs of most labels with most support

    Args:
        node_ids ([type]): [description]
        node_embeddings_2dim ([type]): [description]
        labels_to_node ([type]): [description]
    """
    import matplotlib.pyplot as plt

    markers = ["o", "s", "p", "*"]
    if "NA" in labels_to_node:
        del labels_to_node["NA"]
    #breakpoint()
    labels_to_node = {label: labels_to_node[label] for label in labels_to_node if len(labels_to_node[label]) > 0}
    labels_to_plot = sorted(labels_to_node, key=lambda key: len(labels_to_node[key]), reverse=True)[
        : len(colors) * len(markers)
    ]#[:20]
    # print("ploting these labels", [l, colors[il], len(labels_to_node[l]) for il, l in enumerate(labels_to_plot)])
    x_to_plot = []
    y_to_plot = []
    colors_to_plot = []
    sizes_to_plot = []
    markers_to_plot = []
    #print(labels_to_plot)
    plt.figure()
    #print(" LABEL  COLOR  SIZE   DOTS")
    for i, l in enumerate(labels_to_plot):
        valid_nodes = 0
        if len(labels_to_node) == 0:
            continue
        for node in labels_to_node[l]:
            if node not in node_ids:
                # print("skipping", node)
                continue
            node_idx = node_ids.index(node)
            x_to_plot.append(node_embeddings_2dim[node_idx][0])
            y_to_plot.append(node_embeddings_2dim[node_idx][1])
            if node_sizes is not None:
                sizes_to_plot.append(node_sizes[node_idx])
            else:
                sizes_to_plot.append(50)
            valid_nodes += 1
            colors_to_plot.append(colors[i % len(colors)])
            markers_to_plot.append(markers[i // len(colors)])
        # breakpoint()
        #   print("plotting", l, colors[i % len(colors)], markers[i // len(colors)], len(labels_to_node[l]), valid_nodes)
        # plt.scatter(x_to_plot, y_to_plot, s=sizes_to_plot, c=colors[i], label=l)  # , alpha=0.5)
        sc = plt.scatter(
            x_to_plot,
            y_to_plot,
            s=sizes_to_plot,
            c=colors[i % len(colors)],
            label=l,
            marker=markers[i // len(colors)],
            alpha=0.4,
        )  # , alpha=0.5)
        x_to_plot = []
        y_to_plot = []
        sizes_to_plot = []
    plt.legend('')
    if centroids is not None:
        hq_centroids_mask = [x in hq_centroids for x in range(len(centroids))]
        lq_centroids_mask = [x not in hq_centroids for x in range(len(centroids))]
        # lq_centroids = set(range(len(centroids))) - set(hq_centroids)

        # plt.scatter(
        #    centroids[lq_centroids_mask, 0], centroids[lq_centroids_mask, 1], c="black", label="centroids (LQ)", marker="x"
        # )
        plt.scatter(
            centroids[hq_centroids_mask, 0],
            centroids[hq_centroids_mask, 1],
            c="black",
            label="centroids (HQ)",
            marker="P",
        )

    # for n in node_embeddings:
    # plt.scatter(x_to_plot, y_to_plot, c=colors_to_plot) #, alpha=0.5)
    #plt.legend()
    if outputname is not None:
        print("saving embs plot to {}".format(outputname))
        plt.savefig(outputname, bbox_inches="tight", dpi=400)
    else:
        plt.show()


def cluster_embs(node_embeddings, node_ids, clusteringalgo, kclusters, device="cpu", node_lens=None, seed=0):
    #set_seed(seed)
    if clusteringalgo == "vamb":
        from vamb.cluster import cluster as vamb_cluster
        it = vamb_cluster(
            node_embeddings, node_ids, cuda=(device == "cuda:0")
        )  # , seeds=seeds)  # contig_lens=node_lens)  #
        cluster_to_contig = {i: c for (i, (n, c)) in enumerate(it)}

        # get embs to clusters (cluster to contig has only contig names, not index)
        """cluster_to_embs = {
            c: np.array([node_embeddings[i] for i, n in enumerate(node_ids) if n in cluster_to_contig[c]])
            for c in cluster_to_contig
        }
        cluster_centroids = np.array([cluster_to_embs[c].mean(0) for c in cluster_to_contig])"""
        cluster_centroids = None  # not necessary for now
    else:
        from sklearn.cluster import (
            KMeans,
            DBSCAN,
            AgglomerativeClustering,
            MiniBatchKMeans,
            SpectralClustering,
            Birch,
            OPTICS,
        )
        from sklearn.mixture import GaussianMixture

        if clusteringalgo == "kmeans":
            clustering = KMeans(n_clusters=kclusters, random_state=SEED)
            cluster_labels = clustering.fit_predict(node_embeddings)
            cluster_centroids = clustering.cluster_centers_
        elif clusteringalgo == "kmeansgpu":
            node_embeddings = torch.tensor(node_embeddings).cuda()
            cluster_labels, cluster_centroids = kmeans_pytorch.kmeans(
                X=node_embeddings, num_clusters=kclusters, device=torch.device("cuda:0")
            )
        elif clusteringalgo == "dbscan":
            cluster_labels = DBSCAN(eps=1.1, min_samples=5).fit_predict(node_embeddings)
            # cluster_centroids = clustering.cluster_centers_
            cluster_centroids = None
        elif clusteringalgo == "gmm":
            cluster_model = GaussianMixture(
                n_components=kclusters,
                covariance_type="full",
                max_iter=1000,
                random_state=SEED,
                verbose=2,
                verbose_interval=1,
            ).fit(node_embeddings)
            cluster_labels = cluster_model.predict(node_embeddings)
            cluster_centroids = cluster_model.means_
        elif clusteringalgo == "kmeansconst":
            cluster_labels = KMeansConstrained(
                n_clusters=kclusters, size_min=1, size_max=5, random_state=SEED
            ).fit_predict(node_embeddings)
        elif clusteringalgo == "kmeansbatch":
            kmeans = MiniBatchKMeans(n_clusters=kclusters, random_state=SEED, batch_size=100, init=seed_matrix)
            cluster_labels = kmeans.fit_predict(node_embeddings)
            cluster_centroids = kmeans.cluster_centers_
        elif clusteringalgo == "spectral":
            cluster_labels = SpectralClustering(n_clusters=kclusters, random_state=SEED).fit_predict(node_embeddings)
            cluster_centroids = None
        elif clusteringalgo == "birch":
            cluster_labels = Birch(n_clusters=kclusters).fit_predict(node_embeddings)
            cluster_centroids = None
        elif clusteringalgo == "optics":
            cluster_labels = OPTICS(min_samples=5, cluster_method="xi", n_jobs=-1).fit_predict(node_embeddings)
            cluster_centroids = None
        else:
            print("invalid clustering algorithm")
            return None
        # compare clustering labels with actual labels
        # for each cluster, get majority label, and then P/R
        cluster_to_contig = {i: [] for i in range(kclusters)}
        for il, l in enumerate(cluster_labels):
            # if l not in cluster_to_contig:
            #    cluster_to_contig[l] = []
            cluster_to_contig[l].append(node_ids[il])

    return cluster_to_contig, cluster_centroids


def evaluate_binning(cluster_to_contig, node_to_label, label_to_node, outputclusters=False, contig_sizes=None):
    """Evaluate binning results using contig labels (supervised scenario)

    :param cluster_to_contig: mapping cluster IDs to contig names
    :type cluster_to_contig: dict
    :param node_to_label: mapping contig ids to labels (taxon)
    :type node_to_label: dict
    :param label_to_node: mapping label names to cotnigs
    :type label_to_node: dict
    :param outputclusters: print members of all clusters, defaults to False
    :type outputclusters: bool, optional
    :param contig_sizes: Provide contig sizes for balanced scores, defaults to None
    :type contig_sizes: dict, optional
    """
    # uncomment if you want to plot only the biggest labels
    # labels_to_plot = sorted(label_to_node, key = lambda key: len(label_to_node.get(key, [])), reverse=True)[:6]
    # print(labels_to_plot)
    avg_precision = []
    avg_recall = []
    avg_f1 = []
    cluster_sizes = []
    if contig_sizes is None:
        contig_sizes = {c: 1 for c in node_to_label}
    for c in cluster_to_contig:
        cluster_labels = [node_to_label.get(n) for n in cluster_to_contig[c] if n in node_to_label]
        cluster_counts = {}
        for label in set(cluster_labels):
            cluster_counts[label] = sum(
                [contig_sizes[n] for n in cluster_to_contig[c] if node_to_label.get(n) == label]
            )
        if len(cluster_labels) == 0:  # we do not have labels for any of the elements of this cluster
            continue
        # get majority label:
        # cluster_counter = collections.Counter(cluster_labels)
        # cluster_majority = cluster_counter.most_common(1)
        cluster_majority = max(cluster_counts.items(), key=operator.itemgetter(1))
        # print(cluster_majority)
        # if cluster_majority[0][0] not in labels_to_plot:
        #    continue
        if cluster_majority[0] not in label_to_node:
            print(
                "tie cluster",
                f"cluster {c}, majority: {cluster_majority[0]}, cluster size {len(cluster_to_contig[c])}",
            )
            continue
        # print(f"cluster {c}, majority: {cluster_majority}, cluster size {len(cluster_to_contig[c])}")
        # print(f" {len(label_to_node.get(cluster_majority[0], []))} contigs with this label")
        # avg_precision.append(len(cluster_to_contig) * cluster_majority[0][1]/len(cluster_to_contig[c]))
        # avg_recall.append(len(cluster_to_contig) * cluster_majority[0][1]/len(label_to_node[cluster_majority[0][0]]))
        cluster_size = sum([contig_sizes.get(n, 1) for n in cluster_to_contig[c]])
        cluster_p = cluster_majority[1] / cluster_size
        avg_precision.append(cluster_p)
        cluster_r = cluster_majority[1] / sum([contig_sizes.get(n, 1) for n in label_to_node[cluster_majority[0]]])
        avg_recall.append(cluster_r)
        cluster_f1 = 2 * cluster_p * cluster_r / (cluster_p + cluster_r)
        avg_f1.append(cluster_f1)
        # print(cluster_p, cluster_r)
        if outputclusters:
            print(cluster_to_contig[c])
        cluster_sizes.append(len(cluster_to_contig))
    # print("average precision", sum(avg_precision)/sum(cluster_sizes)/len(avg_precision))
    # print("average recall", sum(avg_recall)/sum(cluster_sizes)/len(avg_recall))
    print(
        "average precision",
        round(sum(avg_precision) / len(avg_precision), 4),
        "average recall",
        round(sum(avg_recall) / len(avg_recall), 4),
        "average f1",
        round(sum(avg_f1) / len(avg_f1), 4),
        "P>0.95 and R>0.9:",
        len([i for i in range(len(avg_recall)) if avg_recall[i] >= 0.9 and avg_precision[i] >= 0.95]),
    )


def read_contigs_scg(ref_file, marker_file, node_names):
    """Read marker stats and return count table

    :param ref_file: path to file with reference markers (e.g. Bacteria.ms)
    :type ref_file: str
    :param marker_file: path to file with markers found on each contig
    :type marker_file: str
    :param node_names: list of node names
    :type node_names: list
    :return: matrix where rows are contigs, columns are markers, values are marker counts
    :rtype: [type]
    """
    ref_marker_genes = read_marker_gene_sets(ref_file)
    contigs_markers = read_contig_genes(marker_file)

    # flatten ref marker gene sets
    ref_marker_genes = [g for m in ref_marker_genes for g in m]
    counts = []
    for contig in node_names:
        counts.append([contigs_markers[contig].get(g, 0) for g in ref_marker_genes])

    return np.array(counts)


def calculate_bin_metrics(results, extra_metrics=False):
    """Calculate overall scores over a set of bins for which we already have scores

    :param results: scores of each bin
    :type results: dict
    :param extra_metrics: Calculate more metrics, defaults to False
    :type extra_metrics: bool, optional
    :return: overall metrics for this bin set
    :rtype: dict
    """
    hq_bins = [bin for bin in results if results[bin]["comp"] >= 90 and results[bin]["cont"] < 5]
    mq_bins = [bin for bin in results if results[bin]["comp"] >= 50 and results[bin]["cont"] < 10]
    metrics = {"hq": hq_bins, "mq": mq_bins, "total": results}
    if extra_metrics and len(results) > 0:
        metrics["avg_comp"] = sum([results[bin]["comp"] for bin in results]) / len(results)
        metrics["avg_cont"] = sum([results[bin]["cont"] for bin in results]) / len(results)
        cont_comp50 = [results[bin]["cont"] for bin in results if results[bin]["comp"] > 50]
        metrics["cont_comp50"] = sum(cont_comp50) / len(cont_comp50) if len(cont_comp50) > 0 else 0
        cont_comp90 = [results[bin]["cont"] for bin in results if results[bin]["comp"] > 90]
        metrics["cont_comp90"] = sum(cont_comp90) / len(cont_comp90) if len(cont_comp90) > 0 else 0
        comp_cont5 = [results[bin]["comp"] for bin in results if results[bin]["cont"] < 5]
        metrics["comp_cont5"] = sum(comp_cont5) / len(comp_cont5) if len(comp_cont5) > 0 else 0
    return metrics


def cluster_eval(
    model,
    dataset,
    logits,
    clustering,
    k,
    loss,
    best_hq,
    best_hq_epoch,
    epoch,
    device,
    clusteringloss=False,
    logger=None,
    use_marker_contigs_only=False,
    seed=0,
):
    """Cluster contig embs and eval with markers

    :param model: Model used to generate embs, save if better than best_hq
    :type model: nn.Module
    :param dataset: dataset object used to train model
    :type dataset: ContigsDataset
    :param logits: tensor with output of model
    :type logits: torch.Tensor
    :param clustering: Type of clustering to be done
    :type clustering: str
    :param k: Number of clusters
    :type k: int
    :param loss: loss (for logging)
    :type loss: [type]
    :param best_hq: Best HQ obtained at this point
    :type best_hq: int
    :param best_hq_epoch: Epoch where best HQ was obtained
    :type best_hq_epoch: int
    :param epoch: Current epoch
    :type epoch: int
    :param device: If using cuda for clustering
    :type device: str
    :param clusteringloss: Compute a clustering loss, defaults to False
    :type clusteringloss: bool, optional
    :param logger: Logger object, defaults to None
    :type logger: [type], optional
    :return: new best HQ and epoch, clustering loss, cluster to contig mapping
    :rtype: list
    """
    import torch
    kmeans_loss = None
    t0_cluster = time.time()
    model.eval()
    #set_seed(seed)
    if torch.is_tensor(logits):
        embeds = logits.detach().numpy()
    else:
        embeds = logits
    if use_marker_contigs_only:
        marker_mask = [n in dataset.contig_markers and len(dataset.contig_markers[n]) > 0 for n in dataset.node_names]
        print("clustering", sum(marker_mask), "markers", len(marker_mask))
        cluster_embeds = embeds[marker_mask]
        cluster_names = np.array(dataset.node_names)[marker_mask]
    else:
        cluster_embeds = embeds
        cluster_names = dataset.node_names
    cluster_to_contig, centroids = cluster_embs(
        cluster_embeds,
        cluster_names,
        clustering,
        # len(dataset.connected),
        k,
        device=device,
        seed=seed,
    )
    contig_to_cluster = {}
    for bin in cluster_to_contig:
        for contig in cluster_to_contig[bin]:
            contig_to_cluster[contig] = bin
    if dataset.labels is not None and len(dataset.labels) > 1:
        # evaluate_binning(
        #    cluster_to_contig,
        #    dataset.node_to_label,
        #    dataset.label_to_node,
        #   contig_sizes={dataset.contig_names[i]: dataset.nodes_len[i][0] for i in range(len(dataset.contig_names))},
        # )
        #calculate_overall_prf(cluster_to_contig, contig_to_cluster, dataset.node_to_label, dataset.label_to_node)
        calculate_sim_between_same_labels(dataset.node_names, embeds,
                                          zip(dataset.edges_src,dataset.edges_dst),
                                          dataset.label_to_node, dataset.node_to_label)
    if dataset.ref_marker_sets is not None:
        results = evaluate_contig_sets(dataset.ref_marker_sets, dataset.contig_markers, cluster_to_contig)
        metrics = calculate_bin_metrics(results)
        logger.info(
            f"HQ: {len(metrics['hq'])}, MQ:, {len(metrics['mq'])} Total bins: {len(metrics['total'])} Best HQ: {best_hq} Best HQ epoch: {best_hq_epoch}"
        )
        # logger.info("HQ:", hq_bins, "avg comp", avg_comp, "avg cont", avg_cont, "avg cont when comp>50", cont_comp50)
        # logger.info(
        #    "HQ {}, avg comp {:.2f}, avg cont {:.2f}, cont when comp>50 {:.2f}, cont when comp>90 {:.2f}, comp when cont<5 {:.2f}".format(
        #        hq_bins, avg_comp, avg_cont, cont_comp50, cont_comp90, comp_cont5
        #    )
        # )
        # print("clustering time:", time.time() - t0_cluster, embeds.shape)
        if len(metrics["hq"]) > best_hq:
            best_hq = len(metrics["hq"])
            best_hq_epoch = epoch
            logger.info("new best checkpoint, saving checkpoint to best_model_hq.pkl")
            torch.save(model.state_dict(), os.path.join(dataset.cache_dir, "best_model_hq.pkl"))

        if clusteringloss:
            centroids = torch.tensor(centroids, device=device)
            # get centroids of hq bins
            # centroids = centroids[hq]
            # get IDs of contigs of HQ bins
            hq_node_ids = [
                dataset.node_names.index(n)
                for bin in metrics["hq"]
                for n in cluster_to_contig[bin]
                # if sum(dataset.contig_markers[n].values()) > 0
            ]
            # count only nodes with SCGs
            # TODO multiply squared dist by completness of cluster
            # cluster_comp = np.array([results[i]["comp"] / 100 for i, r in enumerate(results)])
            # breakpoint()
            cluster_cont = np.array([i for i, r in enumerate(results) if results[i]["cont"] > 10])

            cont_node_ids = [
                dataset.node_names.index(n)
                for bin in cluster_cont
                for n in cluster_to_contig[bin]
                # if sum(dataset.contig_markers[n].values()) > 0
            ]
            # TODO subtract dist of contaminated bins (same num of good bins)
            hq_logits = logits[hq_node_ids]
            # cont_logits = logits[cont_node_ids]
            # breakpoint()
            # https://discuss.pytorch.org/t/k-means-loss-calculation/22041/7
            kmeans_loss_good = ((hq_logits[:, None] - centroids[None]) ** 2).sum(2).min(1)[0].mean()
            # kmeans_loss_bad = ((cont_logits[:, None] - centroids[None]) ** 2).sum(2).min(1)[0].mean()
            # kmeans_loss = -kmeans_loss_bad
            kmeans_loss = kmeans_loss_good
            # kmeans_loss = kmeans_loss_good - kmeans_loss_bad
            # kmeans_loss = ((logits[:, None] - centroids[None]) ** 2).sum(2).min(1)[0].mean()
            logger.info(
                f"Kmeans loss: {kmeans_loss.item()} on {len(hq_logits)} points {len(centroids)} total clusters",
            )
            # loss = kmeans_loss * alpha + (1 - alpha) * loss
    logger.info(
        "Epoch {:05d} | Best HQ: {} | Best epoch {} | Total loss {:.4f}".format(
            epoch,
            best_hq,
            best_hq_epoch,
            loss.detach(),
        )
    )
    return best_hq, best_hq_epoch, kmeans_loss, cluster_to_contig, metrics


def compute_loss_para(adj, device):
    pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


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

def write_bins(contig_to_bin, outputfile):
    with open(outputfile, "w") as f:
        f.write("#\n@Version:0.9.0\n@SampleID:SAMPLEID\n@@SEQUENCEID\tBINID\n")
        for c in contig_to_bin:
            f.write(f"{str(c)}\t{str(contig_to_bin[c])}\n")