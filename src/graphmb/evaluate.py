import sys
import ast
import os
import numpy as np
import scipy
import itertools
import random
from sklearn.metrics.pairwise import cosine_similarity

from graphmb.visualize import run_tsne, plot_embs
from graphmb.utils import run_clustering
# code to run evaluation based on lineage.ms file (Bacteria) and marker_gene_stats.txt file

# Get precicion
def getPrecision(mat, k, s, total):
    sum_k = 0
    for i in range(k):
        max_s = 0
        for j in range(s):
            if mat[i][j] > max_s:
                max_s = mat[i][j]
        sum_k += max_s  
    return sum_k / total


# Get recall
def getRecall(mat, k, s, total, unclassified):
    sum_s = 0
    for i in range(s):
        max_k = 0
        for j in range(k):
            if mat[j][i] > max_k:
                max_k = mat[j][i]
        sum_s += max_k
    return sum_s / (total + unclassified)


def getAveragePrecision(cluster_to_labels, cluster_sizes):
    avg_precision = []
    for c in cluster_to_labels:
        #TP = cluster_to_labels[c][1]; TP+FP + cluster_sizes[c]
        cluster_p = cluster_to_labels[c][1] / cluster_sizes[c]
        avg_precision.append(cluster_p)
    return round(sum(avg_precision) / len(avg_precision), 4)

def getAverageRecall(label_to_cluster, cluster_to_contig, contig_sizes, label_to_node):
    avg_recall = []
    for label in label_to_cluster:
        if label_to_cluster[label][0] == "NA":
            avg_recall.append(0)
        else:
            cluster_r = label_to_cluster[label][1] / sum([contig_sizes.get(n, 1) for n in cluster_to_contig[label_to_cluster[label][0]]])
            avg_recall.append(cluster_r)
    return round(sum(avg_recall) / len(avg_recall), 4)


# Get ARI
def getARI(mat, k, s, N):
    t1 = 0
    for i in range(k):
        sum_k = 0
        for j in range(s):
            sum_k += mat[i][j]
        t1 += scipy.special.binom(sum_k, 2)
    t2 = 0
    for i in range(s):
        sum_s = 0
        for j in range(k):
            sum_s += mat[j][i]
        t2 += scipy.special.binom(sum_s, 2)
    t3 = t1 * t2 / scipy.special.binom(N, 2)
    t = 0
    for i in range(k):
        for j in range(s):
            t += scipy.special.binom(mat[i][j], 2)
    ari = (t - t3) / ((t1 + t2) / 2 - t3)
    return ari


# Get F1-score
def getF1(prec, recall):
    if prec == 0.0 or recall == 0.0:
        return 0.0
    else:
        return 2 * prec * recall / (prec + recall)


def calculate_overall_prf(cluster_to_contig, contig_to_cluster, node_to_label, label_to_node):
    # calculate how many contigs are in the majority class of each cluster
    total_binned = 0
    # convert everything to ids
    labels = list(label_to_node.keys())
    clusters = list(cluster_to_contig.keys())
    n_pred_labels = len(clusters)
    n_true_labels = len(labels)
    ground_truth_count = len(node_to_label)
    # empty matrix
    bins_species = [[0 for x in range(n_true_labels)] for y in range(n_pred_labels)]
    # update with cluster memberships
    for i in contig_to_cluster:
        if i in node_to_label:
            # breakpoint()
            total_binned += 1
            bins_species[clusters.index(contig_to_cluster[i])][labels.index(node_to_label[i])] += 1
    my_precision = getPrecision(bins_species, n_pred_labels, n_true_labels, total_binned)
    my_recall = getRecall(
        bins_species, n_pred_labels, n_true_labels, total_binned, (ground_truth_count - total_binned)
    )
    #my_ari = getARI(bins_species, n_pred_labels, n_true_labels, total_binned)
    my_f1 = getF1(my_precision, my_recall)
    return my_precision, my_recall, my_f1, 0

def calculate_sim_between_same_labels_small(node_names, embeddings, edges, label_to_node, node_to_label):
    # divide sim between node of same label by avg sim between all nodes
    avg_label_sims = {}
    # use only embeddings of nodes with labels
    edges = np.array(edges)
    # calculate sim between all embs
    all_cosine_sim = cosine_similarity(embeddings, embeddings)
    #https://stackoverflow.com/a/69865919
    i = np.ravel_multi_index(np.array(edges).T, all_cosine_sim.shape)
    edge_sim = all_cosine_sim.take(i) # get sim between embs of edges
    edge_sim = edge_sim.sum()
    #all_cosine_sim = np.triu(all_cosine_sim)
    # all sim should not include sims of edges
    all_cosine_sim = all_cosine_sim.sum() - edge_sim
    all_cosine_sim = all_cosine_sim / ((embeddings.shape[0]**2)-len(edges))
    edge_sim /= len(edges)
    for l in label_to_node:
        label_node_idxs = [node_names.index(n) for n in label_to_node[l]]
        label_embs = np.array(embeddings)[label_node_idxs]
        if label_embs.shape[0] > 1: # at least two nodes
            avg_label_sims[l] = ((cosine_similarity(label_embs, label_embs))).mean()
        # pick a random 
    avg = sum(avg_label_sims.values())/len(avg_label_sims.values())      
    #print(round(avg, 4), round(all_cosine_sim, 4))
          #[(x, round(avg_label_sims[x], 4), len(label_to_node[x])) for x in avg_label_sims][:10]])
    return avg, edge_sim, all_cosine_sim


def calculate_sim_between_same_labels_big(node_names, embeddings, edges, label_to_node, node_to_label):
    # divide sim between node of same label by avg sim between all nodes
    avg_label_sims = {}
    # use only embeddings of nodes in edges to make it more efficient
    edges = np.array(edges)
    node_idx_with_edges = np.unique(edges)
    #label_mask = np.array([node_to_label.get(n, "NA") != "NA" for n in node_names])
    all_cosine_sim = cosine_similarity(embeddings[node_idx_with_edges], embeddings[node_idx_with_edges])
    edges_new = np.searchsorted(node_idx_with_edges, edges)
    #https://stackoverflow.com/a/69865919
    i = np.ravel_multi_index(np.array(edges_new).T, all_cosine_sim.shape)
    edge_sim = all_cosine_sim.take(i)
    edge_sim = edge_sim.mean()
    all_cosine_sim = np.triu(all_cosine_sim)
    all_cosine_sim = all_cosine_sim.mean()
    for l in label_to_node:
        label_node_idxs = [node_names.index(n) for n in label_to_node[l] if node_names.index(n) in node_idx_with_edges]
        label_node_idxs = np.searchsorted(node_idx_with_edges, label_node_idxs)
        label_embs = np.array(embeddings)[label_node_idxs]
        if label_embs.shape[0] > 1: # at least two nodes
            avg_label_sims[l] = (np.triu(cosine_similarity(label_embs, label_embs))).mean()
        # pick a random 
    avg = sum(avg_label_sims.values())/len(avg_label_sims.values())      
    #print(round(avg, 4), round(all_cosine_sim, 4))
          #[(x, round(avg_label_sims[x], 4), len(label_to_node[x])) for x in avg_label_sims][:10]])
    return avg, edge_sim, all_cosine_sim


def read_marker_gene_sets(lineage_file):
    """Open file with gene sets for a taxon

    :param lineage_file: path to marker set file from CheckM (Bacteria.ms)
    :type lineage_file: str
    :return: Marker sets
    :rtype: set
    """
    with open(lineage_file, "r") as f:
        lines = f.readlines()
    # consider only single taxon gene set
    sets = lines[1].strip().split("\t")[-1]
    sets = ast.literal_eval(sets)
    return sets


def read_contig_genes(contig_markers):
    """Open file mapping contigs to genes

    :param contig_markers: path to contig markers (marker stats)
    :type contig_markers: str
    :return: Mapping contig names to markers
    :rtype: dict
    """
    contigs = {}
    with open(contig_markers, "r") as f:
        for line in f:
            values = line.strip().split("\t")
            contig_name = values[0]
            # keep only first two elements
            contig_name = "_".join(contig_name.split("_")[:2])
            contigs[contig_name] = {}
            mappings = ast.literal_eval(values[1])
            for contig in mappings:
                for gene in mappings[contig]:
                    if gene not in contigs[contig_name]:
                        contigs[contig_name][gene] = 0
                    # else:
                    #    breakpoint()
                    contigs[contig_name][gene] += 1
                    if len(mappings[contig][gene]) > 1:
                        breakpoint()
    return contigs


def get_markers_to_contigs(marker_sets, contigs):
    """Get marker to contig mapping

    :param marker_sets: Marker sets from CheckM
    :type marker_sets: set
    :param contigs: Contig to marker mapping
    :type contigs: dict
    :return: Marker to contigs list mapping
    :rtype: dict
    """
    marker2contigs = {}
    for marker_set in marker_sets:
        for gene in marker_set:
            marker2contigs[gene] = []
            for contig in contigs:
                if gene in contigs[contig]:
                    marker2contigs[gene].append(contig)
    return marker2contigs


def evaluate_contig_sets(marker_sets, contig_marker_counts, bin_to_contigs):
    """Calculate completeness and contamination for each bin given a set of
       marker gene sets and contig marker counts

    :param marker_sets: reference marker sets, from Bacteria
    :type marker_sets: list
    :param contig_marker_counts: Counts of each gene on each contig
    :type contig_marker_counts: dict
    :param bin_to_contigs: Mapping bins to contigs
    :type bin_to_contigs: dict
    """
    results = {}
    for bin in bin_to_contigs:
        bin_genes = {}
        for contig in bin_to_contigs[bin]:
            if contig not in contig_marker_counts:
                print("missing", contig)
                continue
            for gene in contig_marker_counts[contig]:
                if gene not in bin_genes:
                    bin_genes[gene] = 0
                bin_genes[gene] += contig_marker_counts[contig][gene]
        comp = completeness(marker_sets, set(bin_genes.keys()))
        cont = contamination(marker_sets, bin_genes)
        results[bin] = {"comp": comp, "cont": cont, "genes": bin_genes}
    return results

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


def compute_hq(cluster_stats, comp_th=90, cont_th=5):
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
    return potential_mags, unresolved_contigs

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
    use_labels=False,
    amber=False,
    compute_pospairs=False,
    unresolved=True
):
    reference_markers = dataset.ref_marker_sets
    contig_genes = dataset.contig_markers
    node_to_gt_idx_label = dataset.node_to_label
    gt_idx_label_to_node = dataset.label_to_node
    cluster_to_contig, contig_to_bin, labels, cluster_centroids = run_clustering(X, node_names,
                                                                                 clustering, cuda, k=k)
   
    scores = {"precision": 0, "recall": 0, "f1": 0, "ari": 0, "hq": 0, "mq": 0,
        "n_clusters": len(np.unique(labels)), "unresolved_mags": 0}
    positive_pairs = []
    positive_clusters = []
    if contig_genes is not None and len(contig_genes) > 0:
        cluster_stats = compute_cluster_score(reference_markers, contig_genes, node_names, labels)
        hq, positive_clusters = compute_hq(cluster_stats)
        mq, _ = compute_hq(cluster_stats, comp_th=50,cont_th=10)
        """non_comp, _ = compute_hq(cluster_stats, comp_th=0, cont_th=10)
        all_cont, _ = compute_hq(cluster_stats, comp_th=90, cont_th=1000)
        """
        scores["hq"] = hq
        scores["mq"] = mq
        #TODO avg comp and cont of hq bins
        scores["hq_comp"] = np.mean([cluster_stats[c]["comp"] for c in positive_clusters])
        scores["hq_cont"] = np.mean([cluster_stats[c]["cont"] for c in positive_clusters])
        
        if unresolved:
            unresolved_mags, unresolved_contigs = compute_unresolved(reference_markers=reference_markers,
                contig_genes=contig_genes,
                node_names=node_names,
                node_labels=labels,
                resolved_clusters=positive_clusters)
            scores["unresolved_mags"] = unresolved_mags
            scores["unresolved_contigs"] = len(unresolved_contigs)
            unresolved_contigs_with_scgs = np.array([n for i,n in enumerate(node_names) \
                if labels[i] not in positive_clusters and len(dataset.contig_markers[n]) > 0])
            scores["unresolved_contigs_with_scgs"] = len(unresolved_contigs_with_scgs)
        # print(hq, mq, "incompete but non cont:", non_comp, "cont but complete:", all_cont)
        positive_pairs = None
        if compute_pospairs:
            positive_pairs = get_positive_pairs(node_names, positive_clusters, cluster_to_contig, max_pos_pairs)
    
    # calculate edge similarity metrics
    if node_to_gt_idx_label is not None and len(dataset.labels) > 1:
        #if len(dataset.node_names) < 10_000:
        #    sims = calculate_sim_between_same_labels_small(dataset.node_names, X,
        #                                            list(zip(dataset.edges_src, dataset.edges_dst)),
        #                                            dataset.label_to_node, dataset.node_to_label)
        #elif len([n for n in dataset.node_names if dataset.node_to_label.get(n, "NA") != "NA"]) < 10_000:
        #    sims = calculate_sim_between_same_labels_big(dataset.node_names, X,
        #                                        list(zip(dataset.edges_src, dataset.edges_dst)),
        #                                        dataset.label_to_node, dataset.node_to_label)
        #else:
        sims = 1,1,1
        scores["avg_label_sim"], scores["avg_edge_sim"], scores["avg_total_sim"] = sims
        scores["ratio_labelsim"] = scores["avg_label_sim"]/scores["avg_total_sim"]
        scores["ratio_edgesim"] = scores["avg_edge_sim"]/scores["avg_total_sim"]
        scores["ratio_labeltoedgesim"] = scores["avg_label_sim"]/scores["avg_edge_sim"]

    if amber == True:
        # TODO use p/r/ to get positive_clusters
        import graphmb.amber_eval as amber
        # save to file
        output_bins_filename = dataset.cache_dir + f"/{dataset.name}_temp_contig2bin.tsv"
        amber.write_amber_bins(contig_to_bin, output_bins_filename)
        amber_metrics, bin_counts = amber.amber_eval(os.path.join(dataset.data_dir, dataset.labelsfile), output_bins_filename, ["graphmb"])
        scores["precision_avg_bp"] = amber_metrics["precision_avg_bp"]
        scores["recall_avg_bp"] = amber_metrics["recall_avg_bp"]
        scores["f1_avg_bp"] = amber_metrics["f1_avg_bp"]
        scores["amber_hq"] = bin_counts["> 90% completeness"][1]
        scores["amber_mq"] = bin_counts["> 50% completeness"][0]
    else:
        # calculate edge metrics
        p, r, f1, ari = calculate_overall_prf(
            cluster_to_contig, contig_to_bin, node_to_gt_idx_label, gt_idx_label_to_node
        )
        scores["precision"] = p
        scores["recall"] = r
        scores["f1"] = f1
        scores["ari"] = ari
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
    #print("calc metrics time", datetime.datetime.now() - processing_time)
    return (
        labels,
        scores,
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

def eval_epoch(node_new_features, cluster_mask, weights, args, dataset, epoch, scores, best_metric, 
               best_embs, best_epoch, best_model, target_metric="hq"):
    # used only by vgae model
    #log_to_tensorboard(summary_writer, {"Embs average": np.mean(node_new_features), 'Embs std': np.std(node_new_features) }, step)
    cluster_labels, stats, _, hq_bins = compute_clusters_and_stats(
        node_new_features[cluster_mask], np.array(dataset.node_names)[cluster_mask],
        dataset, clustering=args.clusteringalgo, k=args.kclusters, tsne=args.tsne,
        amber=(args.labels is not None and "amber" in args.labels), cuda=args.cuda,
        compute_pospairs=False, unresolved=True
    )
    
    stats["epoch"] = epoch
    scores.append(stats)
    #logger.info(str(stats))

    #log_to_tensorboard(summary_writer, {"hq": stats["hq"], "mq": stats["mq"]}, step)
    #all_cluster_labels.append(cluster_labels)
    #if dataset.contig_markers is not None and len(dataset.contig_markers) > 0 and stats["hq"] > best_metric:
    #    best_metric, best_embs, best_epoch, best_model = stats["hq"], node_new_features, epoch, weights
        #best_model = th.gnn_model
        #save_model(args, e, th, th_vae)
    if stats[target_metric] > best_metric:
        best_metric, best_embs, best_epoch, best_model = stats[target_metric], node_new_features, epoch, weights

    return best_metric, best_embs, best_epoch, scores, best_model, cluster_labels


def eval_epoch_cluster(logger, summary_writer, node_new_features, cluster_mask, best_hq,
               step, args, dataset, epoch, tsne, cluster=True):
    
    if cluster:
        #log_to_tensorboard(summary_writer, {"Embs average": np.mean(node_new_features), 'Embs std': np.std(node_new_features) }, step)
        tsne_path = os.path.join(args.outdir, f"{args.outname}_tsne_clusters_epoch_{epoch}.png")
        cluster_labels, stats, positive_pairs, hq_bins = compute_clusters_and_stats(
            node_new_features[cluster_mask], np.array(dataset.node_names)[cluster_mask],
            dataset, clustering=args.clusteringalgo, k=args.kclusters, tsne=tsne, tsne_path=tsne_path, max_pos_pairs=None,
            amber=(args.labels is not None and "amber" in args.labels), cuda=args.cuda,
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

    #log_to_tensorboard(summary_writer, {"hq_bins": stats["hq"], "mq_bins": stats["mq"]}, step)
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

def main():
    sets = read_marker_gene_sets(sys.argv[1])
    markers = read_contig_genes(sys.argv[2])
    #print("Single contig bin eval")
    if len(sys.argv) < 4:
        print("Single contig bin eval")
        single_contig_bins = {c: [c] for c in markers}
        results = evaluate_contig_sets(sets, markers, single_contig_bins)
        total_hq = 0
        for bin in results:
            if results[bin]["comp"] > 90 and results[bin]["cont"] < 5:
                print(bin, results[bin]["comp"], results[bin]["cont"])
                total_hq += 1
        print("Total HQ", total_hq)
        # for bin in markers:
        #    print(bin)
        #    print(completeness(sets, set(markers[bin].keys())))
        #    print(contamination(sets, markers[bin]))
    else:
        cluster_to_contigs = {}
        clusters_file = sys.argv[3]
        with open(clusters_file, "r") as f:
            for line in f:
                if line.startswith("@") or line.startswith("#"):
                    continue
                values = line.strip().split()
                contig, binname = values[0], values[1]
                if binname not in cluster_to_contigs:
                    cluster_to_contigs[binname] = []
                cluster_to_contigs[binname].append(contig)
        results = evaluate_contig_sets(sets, markers, cluster_to_contigs)
        # print(results)

        total_hq = 0
        for bin in results:
            if results[bin]["comp"] > 90 and results[bin]["cont"] < 5:
                print(bin, results[bin]["comp"], results[bin]["cont"])
                total_hq += 1
        print("Total HQ", total_hq)
        if len(sys.argv) > 4:
            import graphmb.amber_eval as amber
            scores = {}
            amber_metrics, bin_counts = amber.amber_eval(sys.argv[4], clusters_file, ["graphmb"])
            scores["precision_avg_bp"] = amber_metrics["precision_avg_bp"]
            scores["recall_avg_bp"] = amber_metrics["recall_avg_bp"]
            scores["f1_avg_bp"] = amber_metrics["f1_avg_bp"]
            print(scores)
        # count how many contigs per marker gene
    scg_counts = {}
    for marker_set in sets:
        for gene in marker_set:
            scg_counts[gene] = 0
            for contig in markers:
                if gene in markers[contig]:
                    scg_counts[gene] += markers[contig][gene]

    #print(scg_counts)
    quartiles = np.percentile(list(scg_counts.values()), [25, 50, 75])
    print("candidate k0s", sorted(set([k for k in scg_counts.values() if k >= quartiles[2]])))

    # TODO convert SCGs to features


if __name__ == "__main__":
    main()
