import itertools
import argparse
import shutil
import random
import time
import logging
from datetime import datetime
from collections import Counter
import numpy as np
import sys
import copy
import pickle
import shutil
import pdb
import dgl
import torch
import torch.nn as nn
import networkx as nx

import os
from graphmb.contigsdataset import AssemblyDataset, DGLAssemblyDataset
from pathlib import Path
import scipy.stats as stats
from graphmb.evaluate import (
    evaluate_contig_sets,
    calculate_overall_prf,
)
from graphmb.graphsage_unsupervised import train_graphsage, SAGE
from graphmb.graph_functions import (
    plot_embs,
    cluster_embs,
    evaluate_binning,
    calculate_bin_metrics,
    draw_nx_graph,
    set_seed,
)
import vaegbin
from graphmb.version import __version__

SEED = 0


def run_tsne(embs, dataset, cluster_to_contig, hq_bins, centroids=None):
    from sklearn.manifold import TSNE

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
        all_embs = tsne.fit_transform(torch.cat((torch.tensor(embs), torch.tensor(centroids)), dim=0))
        centroids_2dim = all_embs[embs.shape[0] :]
        node_embeddings_2dim = all_embs[: embs.shape[0]]
    else:
        centroids_2dim = None
        node_embeddings_2dim = tsne.fit_transform(torch.tensor(embs))
    return node_embeddings_2dim, centroids_2dim


def draw(dataset, node_to_label, label_to_node, cluster_to_contig, outname, graph=None):
    # properties of all nodes
    nodeid_to_label = {i: node_to_label.get(n, "NA") for i, n in enumerate(dataset.node_names)}
    contig_lens = {i: dataset.node_lengths[i] for i in range(len(dataset.node_names))}
    nodes_titles = {
        i: str(dataset.node_names[i]) + "<br>Length: " + str(contig_lens[i]) for i in range(len(dataset.contig_names))
    }
    if dataset.depth is not None:
        nodes_titles = {
            i: nodes_titles[i] + "<br>Depth: " + ", ".join(["{:.4}".format(x) for x in dataset.nodes_depths[i]])
            for i in range(len(dataset.contig_names))
        }
    if cluster_to_contig:
        contig_to_cluster = {contig: cluster for cluster, contigs in cluster_to_contig.items() for contig in contigs}
        nodes_titles = {
            i: nodes_titles[i] + "<br>Cluster: " + str(contig_to_cluster[n])
            for i, n in enumerate(dataset.contig_names)
        }

    # convert DGL graph to networkx
    nx_graph = graph.cpu().to_networkx(edge_attrs=["weight"]).to_undirected()
    connected_comp = [c for c in sorted(nx.connected_components(nx_graph), key=len, reverse=True) if len(c) > 0]
    # TODO: draw connected components to separate files
    for i in range(10, 50):
        # without_largest_comp = [item for sublist in connected_comp[10:110] for item in sublist if len(sublist) > 2]
        this_comp = connected_comp[i]
        nx_graph = nx.subgraph(nx_graph, this_comp)

        draw_nx_graph(
            nx_graph,
            nodeid_to_label,
            label_to_node,
            outname + "_" + str(i),
            contig_sizes=contig_lens,
            node_titles=nodes_titles,
        )


def write_embs(embs, node_names, outname):
    # write embs as dict node_name: embs
    embs_dict = {node_names[i]: embs[i] for i in range(len(embs))}
    with open(outname, "wb") as f:
        pickle.dump(embs_dict, f)


def write_edges(graph, outname):
    with open(outname, "w") as graphf:
        for e in zip(graph.edges()[0], graph.edges()[1]):
            graphf.write(str(e[0].item()) + "\t" + str(e[1].item()) + "\n")


def check_dirs(args, use_features=True):
    """Check if files necessary to run exist, other wise print message and exit"""
    if args.outdir is None:
        if args.assembly is None:
            print("Please specify assembly path or outdir with --assembly or --outdir")
            exit()
        else:
            args.outdir = args.assembly
    else:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # check if other dirs exists
    if not os.path.exists(os.path.join(args.assembly, args.graph_file)):
        print(f"Assembly Graph file {args.graph_file} not found")
        exit()
    if not os.path.exists(os.path.join(args.assembly, args.features)) or not use_features:
        # needs assembly files to calculate features
        if not os.path.exists(os.path.join(args.assembly, args.assembly_name)):
            print(f"Assembly {args.assembly_name} not found")
            exit()
        if not os.path.exists(os.path.join(args.assembly, args.depth)):
            print(f"Depth file {args.depth} not found")
            exit()


def get_activation(args):
    # pick activation function
    if args.activation == "prelu":
        activation = nn.PReLU(args.hidden)
    elif args.activation == "relu":
        activation = nn.ReLU()
    elif args.activation == "tanh":
        activation = nn.Tanh()
    elif args.activation == "sigmoid":
        activation = nn.Sigmoid()
    elif args.activation == "lrelu":
        activation = nn.LeakyReLU()
    return activation


def run_graphmb(dataset, args, device, logger):
    activation = get_activation(args)
    model = SAGE(
        dataset.graph.ndata["feat"].shape[1],
        args.hidden,
        args.embsize,
        args.layers,
        activation,
        args.dropout,
        agg=args.aggtype,
    )
    model = model.to(device)

    if model is not None:
        model = model.to(device)

    logging.info(model)
    if (
        dataset.assembly.ref_marker_sets is not None
        and args.clusteringalgo is not None
        and not args.skip_preclustering
    ):
        # cluster using only input features
        logger.info("pre train clustering:")
        pre_cluster_to_contig, centroids = cluster_embs(
            dataset.graph.ndata["feat"].detach().cpu().numpy(),
            dataset.assembly.node_names,
            args.clusteringalgo,
            args.kclusters,
            device=device,
            # node_lens=np.array([c[0] for c in dataset.assembly.nodes_len]),
            node_lens=np.array(dataset.assembly.node_lengths),
        )
        results = evaluate_contig_sets(
            dataset.assembly.ref_marker_sets, dataset.assembly.contig_markers, pre_cluster_to_contig
        )
        calculate_bin_metrics(results, logger=logger)

    best_train_embs, best_model, last_train_embs, last_model = train_graphsage(
        dataset,
        model,
        batch_size=args.batchsize,
        fan_out=args.fanout,
        num_negs=args.negatives,
        neg_share=False,
        num_epochs=args.epoch,
        lr=args.lr,
        k=args.kclusters,
        clusteringalgo=args.clusteringalgo,
        print_interval=args.print,
        loss_weights=(not args.no_loss_weights),
        sample_weights=(not args.no_sample_weights),
        logger=logger,
        device=device,
        epsilon=args.early_stopping,
        evalepochs=args.evalepochs,
    )
    return best_train_embs, best_model, last_train_embs, last_model


def write_bins(args, dataset, cluster_to_contig, logger):
    bin_dir = Path(args.outdir + "/{}_bins/".format(args.outname))
    bin_dir.mkdir(parents=True, exist_ok=True)
    [f.unlink() for f in bin_dir.glob("*.fa") if f.is_file()]
    clustered_contigs = set()
    multi_contig_clusters = 0
    logger.info(f"{len(cluster_to_contig)} clusters")
    short_contigs = set()
    skipped_clusters = 0
    for c in cluster_to_contig:
        cluster_size = sum([dataset.node_lengths[dataset.node_names.index(contig)] for contig in cluster_to_contig[c]])
        if cluster_size < args.minbin:
            # print("skipped small cluster", len(cluster_to_contig[c]), "contig")
            for contig in cluster_to_contig[c]:
                short_contigs.add(contig)
            skipped_clusters += 1
            continue
        multi_contig_clusters += 1
        with open(bin_dir / f"{c}.fa", "w") as binfile:
            for contig in cluster_to_contig[c]:
                binfile.write(">" + contig + "\n")
                binfile.write(dataset.node_seqs[contig] + "\n")
                clustered_contigs.add(contig)
        # print("multi cluster", c, "size", cluster_size, "contigs", len(cluster_to_contig[c]))
    logger.info("skipped {} clusters".format(skipped_clusters))
    single_clusters = multi_contig_clusters
    left_over = set(dataset.node_names) - clustered_contigs - short_contigs
    for c in left_over:
        if c not in clustered_contigs and len(dataset.node_seqs[c]) > args.minbin:

            with open(bin_dir / f"{single_clusters}.fna", "w") as binfile:
                binfile.write(">" + c + "\n")
                binfile.write(dataset.node_seqs[c] + "\n")
                single_clusters += 1
            # print("contig", single_clusters, "size", len(dataset.contig_seqs[c]))
    logger.info(f"wrote {single_clusters} clusters {multi_contig_clusters} >= #contig {args.mincomp}")


def run_post_processing(final_embs, args, logger, dataset, device, label_to_node, node_to_label):
    if "cluster" in args.post or "kmeans" in args.post:
        logger.info("clustering embs with {} ({})".format(args.clusteringalgo, args.kclusters))
        # train_embs = last_train_embs

        if args.clusteringalgo is False:
            args.clusteringalgo = "kmeans"
        if not isinstance(final_embs, np.ndarray):
            final_embs = final_embs.numpy()
            if args.cuda:
                final_embs = final_embs.cpu()

            # last_train_embs should already be detached and on cpu
        best_cluster_to_contig, best_centroids = cluster_embs(
            final_embs,
            dataset.node_names,
            args.clusteringalgo,
            # len(dataset.connected),
            args.kclusters,
            device=device,
        )
        cluster_sizes = {}
        for c in best_cluster_to_contig:
            cluster_size = sum([len(dataset.node_seqs[contig]) for contig in best_cluster_to_contig[c]])
            cluster_sizes[c] = cluster_size
        best_contig_to_bin = {}
        for bin in best_cluster_to_contig:
            for contig in best_cluster_to_contig[bin]:
                best_contig_to_bin[contig] = bin
        # run for best epoch only
        if args.markers is not None:
            total_hq = 0
            total_mq = 0
            results = evaluate_contig_sets(dataset.ref_marker_sets, dataset.contig_markers, best_cluster_to_contig)
            hq_bins = set()
            for binid in results:
                if results[binid]["comp"] > 90 and results[binid]["cont"] < 5:
                    contig_labels = [dataset.node_to_label.get(node, 0) for node in best_cluster_to_contig[binid]]
                    labels_count = Counter(contig_labels)
                    logger.info(
                        f"{binid}, {round(results[binid]['comp'],4)}, {round(results[binid]['cont'],4)}, "
                        f"{len(best_cluster_to_contig[binid])} {labels_count}"
                    )
                    hq_bins.add(binid)
                    total_hq += 1
                if results[binid]["comp"] > 50 and results[binid]["cont"] < 10:
                    total_mq += 1
            logger.info("Total HQ {}".format(total_hq))
            logger.info("Total MQ {}".format(total_mq))
        contig_lens = {dataset.node_names[i]: dataset.node_lengths[i] for i in range(len(dataset.node_names))}
        if len(dataset.labels) > 1:
            evaluate_binning(best_cluster_to_contig, node_to_label, label_to_node, contig_sizes=contig_lens)
            # calculate overall P/R/F
            calculate_overall_prf(best_cluster_to_contig, best_contig_to_bin, node_to_label, label_to_node)
            calculate_overall_prf(
                {
                    cluster: best_cluster_to_contig[cluster]
                    for cluster in best_cluster_to_contig
                    if cluster_sizes[cluster] > args.minbin
                },
                {
                    contig: best_contig_to_bin[contig]
                    for contig in best_contig_to_bin
                    if cluster_sizes[best_contig_to_bin[contig]] > args.minbin
                },
                node_to_label,
                label_to_node,
            )
        if "writebins" in args.post:
            logger.info(f"writing bins to {args.outdir}/{args.outname}_bins/")
            write_bins(args, dataset, best_cluster_to_contig, logger)
        if "contig2bin" in args.post:
            # invert cluster_to_contig
            logging.info("Writing contig2bin to {}/{}".format(args.outdir, args.outname))
            with open(args.outdir + f"/{args.outname}_best_contig2bin.tsv", "w") as f:
                for c in best_contig_to_bin:
                    f.write(f"{str(c)}\t{str(best_contig_to_bin[c])}\n")

    # plot tsne embs
    if "tsne" in args.post:
        node_embeddings_2dim, centroids_2dim = run_tsne(final_embs, dataset, best_cluster_to_contig, hq_bins)
        plot_embs(
            dataset.node_names,
            node_embeddings_2dim,
            label_to_node,
            centroids=centroids_2dim,
            hq_centroids=hq_bins,
            node_sizes=None,
            outputname=args.outdir + args.outname + "_tsne_clusters.png",
        )

        # node_sizes=[dataset.nodes_len[i][0] * 100 for i in range(len(dataset.contig_names))],
    if "draw" in args.post:
        print("drawing graph")
        draw(
            dataset,
            node_to_label,
            label_to_node,
            best_cluster_to_contig,
            args.outdir + args.outname + "_graph.png",
            graph=graph,
        )

    if "edges" in args.post:
        logger.info(f"writing edges to {args.outdir + args.outname}_edges")
        write_edges(graph, args.outdir + args.outname + "_edges")

    if "writeembs" in args.post:
        logger.info("writing best and last embs to {}".format(args.outdir))
        write_embs(final_embs, dataset.node_names, os.path.join(args.outdir, f"{args.outname}_best_embs.pickle"))
        # write_embs(best_train_embs, dataset.node_names, os.path.join(args.outdir, f"{args.outname}_last_embs.pickle"))


def main():
    parser = argparse.ArgumentParser(description="Train graph embedding model")
    # input files
    parser.add_argument("--assembly", type=str, help="Assembly base path", required=False)
    parser.add_argument("--assembly_name", type=str, help="File name with contigs", default="assembly.fasta")
    parser.add_argument("--graph_file", type=str, help="File name with graph", default="assembly_graph.gfa")
    parser.add_argument(
        "--edge_threshold", type=float, help="Remove edges with weight lower than this (keep only >=)", default=None
    )
    parser.add_argument("--depth", type=str, help="Depth file from jgi", default="assembly_depth.txt")
    parser.add_argument(
        "--features", type=str, help="Features file mapping contig name to features", default="features.tsv"
    )
    parser.add_argument("--labels", type=str, help="File mapping contig to label", default=None)
    parser.add_argument("--embs", type=str, help="No train, load embs", default=None)

    # model specification
    parser.add_argument("--model_name", type=str, help="only sage for now", default="sage_lstm")
    parser.add_argument(
        "--activation", type=str, help="Activation function to use(relu, prelu, sigmoid, tanh)", default="relu"
    )
    parser.add_argument("--layers", type=int, help="Number of layers of the GNN", default=3)
    parser.add_argument("--hidden", type=int, help="Dimension of hidden layers of GNN", default=512)
    parser.add_argument("--embsize", type=int, help="Output embedding dimension of GNN", default=64)
    parser.add_argument("--batchsize", type=int, help="batchsize to train the GNN", default=0)
    parser.add_argument("--dropout", type=float, help="dropout of the GNN", default=0.0)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.00005)
    parser.add_argument("--clusteringalgo", help="clustering algorithm", default="vamb")
    parser.add_argument("--kclusters", help="Number of clusters (only for some clustering methods)", default=None)
    # GraphSAGE params
    parser.add_argument("--aggtype", help="Aggregation type for GraphSAGE (mean, pool, lstm, gcn)", default="lstm")
    parser.add_argument("--negatives", help="Number of negatives to train GraphSAGE", default=1, type=int)
    parser.add_argument(
        "--fanout", help="Fan out, number of positive neighbors sampled at each level", default="10,25"
    )
    # other training params
    parser.add_argument("--epoch", type=int, help="Number of epochs to train model", default=100)
    parser.add_argument("--print", type=int, help="Print interval during training", default=10)
    parser.add_argument("--evalepochs", type=int, help="Epoch interval to run eval", default=10)
    parser.add_argument("--kmer", default=4)
    parser.add_argument("--usekmer", help="Use kmer features", action="store_true")
    parser.add_argument("--clusteringloss", help="Train with clustering loss", action="store_true")
    parser.add_argument("--no_loss_weights", action="store_false", help="Using edge weights for loss (positive only)")
    parser.add_argument("--no_sample_weights", action="store_false", help="Using edge weights to sample negatives")
    parser.add_argument(
        "--early_stopping",
        type=float,
        help="Stop training if delta between last two losses is less than this",
        default="0.1",
    )
    # data processing
    parser.add_argument("--mincontig", type=int, help="Minimum size of input contigs", default=1000)
    parser.add_argument("--minbin", type=int, help="Minimum size of clusters in bp", default=200000)
    parser.add_argument("--mincomp", type=int, help="Minimum size of connected components", default=1)
    parser.add_argument("--randomize", help="Randomize graph", action="store_true")
    parser.add_argument("--no_edges", help="Add only self edges", action="store_true")
    parser.add_argument("--read_embs", help="Read embeddings from file", action="store_true")
    parser.add_argument("--reload", help="Reload data", action="store_true")

    parser.add_argument("--markers", type=str, help="File with precomputed checkm results to eval", default=None)
    parser.add_argument("--post", help="Output options", default="cluster_contig2bins_writeembs_writebins")
    parser.add_argument("--skip_preclustering", help="Use precomputed checkm results to eval", action="store_true")
    parser.add_argument("--outname", "--outputname", help="Output (experiment) name", default="")
    parser.add_argument("--cuda", help="Use gpu", action="store_true")
    parser.add_argument("--vamb", help="Run vamb instead of loading features file", action="store_true")
    parser.add_argument("--vambdim", help="VAE latent dim", default=32)
    parser.add_argument("--numcores", help="Number of cores to use", default=1, type=int)
    parser.add_argument(
        "--outdir", "--outputdir", help="Output dir (same as input assembly dir if not defined", default=None
    )
    parser.add_argument("--assembly_type", help="flye or spades", default="flye")
    parser.add_argument("--seed", help="Set seed", default=1, type=int)
    parser.add_argument("--version", "-v", help="Print version and exit", action="store_true")
    args = parser.parse_args()

    if args.version:
        print(f"GraphMB {__version__}")
        exit(0)

    check_dirs(args)

    # set up logging
    now = datetime.now()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logfile = os.path.join(args.outdir, now.strftime("%Y%m%d-%H%M%S") + "{}_output.log".format(args.outname))
    output_file_handler = logging.FileHandler(logfile)
    print("logging to {}".format(logfile))

    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.info(args)
    logger.addHandler(stdout_handler)
    logging.getLogger("matplotlib.font_manager").disabled = True

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    logger.info(f"Running GraphMB {__version__}")

    # setup cuda and cpu
    logger.info("using cuda: {}".format(str(args.cuda)))
    device = "cuda:0" if args.cuda else "cpu"
    logger.info(f"cuda available: {(device == 'cuda:0')} using {device}")
    torch.set_num_threads(args.numcores)

    logger.info("setting seed to {}".format(args.seed))
    set_seed(args.seed)

    # specify data properties for caching
    if args.features is None:
        if args.assembly != "":
            features_dir = os.path.join(args.assembly, "features.tsv")
        else:
            features_dir = os.path.join(args.outdir, "features.tsv")
    else:
        features_dir = os.path.join(args.assembly, args.features)

    # create assembly object
    dataset = AssemblyDataset(
        logger,
        args.assembly,
        args.assembly_name,
        args.graph_file,
        args.depth,
        args.markers,
        args.labels,
        features_dir,
        args.outdir,
        min_contig_length=args.mincontig,
    )
    if dataset.check_cache() and not args.reload:
        dataset.read_cache()
    else:
        check_dirs(args, use_features=False)
        dataset.read_assembly()
    dataset.read_scgs()
    # k can be user defined or dependent on the dataset
    if args.kclusters is None:
        args.kclusters = len(dataset.labels)
    args.kclusters = int(args.kclusters)

    # filter graph by components
    # dataset.connected = [c for c in dataset.connected if len(c) >= args.mincomp]

    #### select features to use
    # this code wont be used for now but next version will train AE from these features
    # zscore Kmer features (kmer are already loaded from reading the dataset)
    # dataset.nodes_kmer = torch.FloatTensor(stats.zscore(dataset.nodes_kmer, axis=0))

    vamb_emb_exists = os.path.exists(features_dir)
    if args.vamb or not vamb_emb_exists:
        logger.info("running VAMB...")
        vamb_outdir = os.path.join(args.outdir, "vamb_out{}/".format(args.vambdim))
        dataset.run_vamb(vamb_outdir, args.cuda, args.vambdim)
    dataset.read_features()

    # graph transformations
    # Filter edges according to weight (could be from read overlap count or depth sim)
    # if args.no_edges:
    #    dataset.filter_edges(10e6)
    # elif args.read_edges or args.depth:
    # if args.edge_threshold is not None:
    #    dataset.filter_edges(int(args.edge_threshold))

    if args.embs is not None:  # no training, just run post processing
        emb_file = args.embs
        with open(emb_file, "rb") as embsf:
            best_embs_dict = pickle.load(embsf)
            best_train_embs = np.array([best_embs_dict[i] for i in dataset.node_names])
    # else:
    #

    # DGL specific code
    elif args.model_name == "sage_lstm":
        dgl_dataset = DGLAssemblyDataset(dataset)
        # initialize empty features vector
        nodes_data = torch.FloatTensor(len(dataset.node_names), 0)
        # if args.usekmer:
        #    dataset.nodes_data = torch.cat((dataset.nodes_data, dataset.nodes_kmer), dim=1)
        # if args.depth is not None:
        #    dataset.nodes_data = torch.cat((dataset.nodes_data, dataset.nodes_depths), dim=1)
        if args.features is not None:  # append embs
            node_embs = torch.FloatTensor(dataset.node_embs)
            nodes_data = torch.cat((nodes_data, node_embs), dim=1)
        dgl_dataset.graph.ndata["feat"] = nodes_data
        # dataset.graph.ndata["len"] = torch.Tensor(dataset.nodes_len)

        # All nodes have a self loop
        # dataset.graph = dgl.remove_self_loop(dataset.graph)
        # diff_edges = len(dataset.graph.edata["weight"])
        # dataset.graph = dgl.add_self_loop(dataset.graph)

        # max_weight = dataset.graph.edata["weight"].max().item()
        # dataset.graph.edata["weight"][diff_edges:] = max_weight
        dgl_dataset.graph.edata["weight"] = dgl_dataset.graph.edata["weight"].float()
        graph = dgl_dataset[0]
        logger.info(graph)
        graph = graph.to(device)

        model = None
        if args.embs is None and args.read_embs is False:
            best_train_embs, best_model, last_train_embs, last_model = run_graphmb(dgl_dataset, args, device, logger)
            emb_file = args.outdir + f"/{args.outname}_train_embs.pickle"

        if model is None:
            best_train_embs = graph.ndata["feat"]
            last_train_embs = graph.ndata["feat"]
    elif args.model_name in ("sage", "gcn", "gat"):
        # TODO implement repeats and grid search
        best_train_embs = vaegbin.run_gnn(dataset, args)

    run_post_processing(best_train_embs, args, logger, dataset, device, dataset.label_to_node, dataset.node_to_label)


if __name__ == "__main__":
    main()
