
import time
import os
import networkx as nx
import numpy as np
import operator

#import tensorflow

from graphmb.evaluate import read_contig_genes, read_marker_gene_sets, evaluate_contig_sets
from graphmb.contigsdataset import get_kmer_to_id, count_kmers
from graphmb.utils import SEED
# import torch

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
