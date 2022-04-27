import sys
import ast
import numpy as np
import scipy
import operator

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


def calculate_overall_prf(cluster_to_contig, contig_to_cluster, node_to_label, label_to_node, contig_sizes):
    # calculate how many contigs are in the majority class of each cluster
    total_binned = 0
    # convert everything to ids
    labels = list(label_to_node.keys())
    clusters = list(cluster_to_contig.keys())
    n_pred_labels = len(clusters)
    n_true_labels = len(labels)
    ground_truth_count = len(node_to_label)
    cluster_to_labels = {}
    cluster_sizes = {}
    #bins_species = [[0 for x in range(n_true_labels)] for y in range(n_pred_labels)]
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
        cluster_to_labels[c] = max(cluster_counts.items(), key=operator.itemgetter(1))
        cluster_sizes[c] = sum([contig_sizes[n] for n in cluster_to_contig[c]])
    #for i in contig_to_cluster:
    #    if i in node_to_label:
            # breakpoint()
    #        total_binned += 1
    #        bins_species[clusters.index(contig_to_cluster[i])][labels.index(node_to_label[i])] += 1

    #precision = getPrecision(bins_species, n_pred_labels, n_true_labels, total_binned)
    #recall = getRecall(
    #    bins_species, n_pred_labels, n_true_labels, total_binned, (ground_truth_count - total_binned)
    #)
    #breakpoint()
    label_to_cluster = {}
    for label in labels:
        clusters_with_label = {c: cluster_to_labels[c] for c in cluster_to_labels if cluster_to_labels[c][0] == label}
        if len(clusters_with_label) == 0:
            label_to_cluster[label] = ("NA", 0)
        else:
            label_to_cluster[label] = max(clusters_with_label.items(), key=operator.itemgetter(1))
            label_to_cluster[label] = (label_to_cluster[label][0], label_to_cluster[label][1][1])
    avg_precision = getAveragePrecision(cluster_to_labels, cluster_sizes)
    avg_recall = getAverageRecall(label_to_cluster, cluster_to_contig, contig_sizes, label_to_node)   #ari = getARI(bins_species, n_pred_labels, n_true_labels, total_binned)
    f1 = getF1(avg_precision, avg_recall)
    print("### Evaluation {} cluster/{} labels:".format(n_pred_labels, n_true_labels))
    #print("### Precision = %0.4f  Recall = %0.4f  F1 = %0.4f ARI = %0.4f" % (precision, recall, f1, ari))
    print("### Precision = %0.4f  Recall = %0.4f  F1 = %0.4f" % (avg_precision, avg_recall, f1, ))
    return {"precision": avg_precision, "recall": avg_recall, "f1": f1} #, "ari": ari}


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
    """Calculate completeness and contamination for each bin given a set of marker gene sets and contig marker counts

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


def completeness(marker_sets, marker_genes_bin):
    numerator = 0.0
    for marker_set in marker_sets:
        common = marker_set & marker_genes_bin
        if len(common) > 0:
            numerator += len(common) / len(marker_set)
    return 100 * (numerator / len(marker_sets))


def contamination(marker_sets, marker_count_bin):
    numerator = 0.0
    for i, marker_set in enumerate(marker_sets):
        set_total = 0.0
        for g in marker_set:
            if g in marker_count_bin and marker_count_bin[g] > 0:
                set_total += marker_count_bin[g] - 1.0
        numerator += set_total / len(marker_set)
    return 100.0 * (numerator / len(marker_sets))


def main():
    sets = read_marker_gene_sets(sys.argv[1])
    markers = read_contig_genes(sys.argv[2])
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
    if len(sys.argv) > 3:
        cluster_to_contigs = {}
        clusters_file = sys.argv[3]
        with open(clusters_file, "r") as f:
            for line in f:
                values = line.strip().split("\t")
                if values[0] not in cluster_to_contigs:
                    cluster_to_contigs[values[0]] = []
                cluster_to_contigs[values[0]].append(values[1])
        results = evaluate_contig_sets(sets, markers, cluster_to_contigs)
        # print(results)

        total_hq = 0
        for bin in results:
            if results[bin]["comp"] > 90 and results[bin]["cont"] < 5:
                print(bin, results[bin]["comp"], results[bin]["cont"])
                total_hq += 1
        print("Total HQ", total_hq)
    # count how many contigs per marker gene
    scg_counts = {}
    for marker_set in sets:
        for gene in marker_set:
            scg_counts[gene] = 0
            for contig in markers:
                if gene in markers[contig]:
                    scg_counts[gene] += markers[contig][gene]

    print(scg_counts)
    quartiles = np.percentile(list(scg_counts.values()), [25, 50, 75])
    print("candidate k0s", sorted(set([k for k in scg_counts.values() if k >= quartiles[2]])))

    # TODO convert SCGs to features


if __name__ == "__main__":
    main()
