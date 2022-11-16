
import networkx as nx
import numpy as np
#import tensorflow

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
