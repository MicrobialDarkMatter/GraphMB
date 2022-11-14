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
import os
import copy
import pickle
import shutil
import pdb
import dgl
import torch
import torch.nn as nn
import networkx as nx

print(os.environ["DGLBACKEND"])
from graphmb.contigsdataset import ContigsDataset
from pathlib import Path
import scipy.stats as stats
from graphmb.evaluate import (
    read_marker_gene_sets,
    read_contig_genes,
    evaluate_contig_sets,
    get_markers_to_contigs,
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
from graphmb.version import __version__
from vamb.vamb_run import run as run_vamb

SEED = 0
BACTERIA_MARKERS = "data/Bacteria.ms"


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
    parser.add_argument("--model", type=str, help="only sage for now", default="sage")
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
    parser.add_argument("--contignodes", help="Use contigs as nodes instead of edges", action="store_true")
    parser.add_argument("--seed", help="Set seed", default=1, type=int)
    parser.add_argument("--version", "-v", help="Print version and exit", action="store_true")
    args = parser.parse_args()

    if args.version:
        print(f"GraphMB {__version__}")
        exit(0)

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
    if not os.path.exists(os.path.join(args.assembly, args.features)) and \
        not os.path.exists(os.path.join(args.outdir, args.features)):
        # needs assembly files to calculate features
        if not os.path.exists(os.path.join(args.assembly, args.assembly_name)):
            print(f"Assembly {args.assembly_name} not found")
            exit()
        if not os.path.exists(os.path.join(args.assembly, args.depth)):
            print(f"Depth file {args.depth} not found")
            exit()

    print("setting seed to {}".format(args.seed))
    set_seed(args.seed)
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

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    logger.info(f"Running GraphMB {__version__}")
    logging.getLogger("matplotlib.font_manager").disabled = True

    # setup cuda and cpu
    logging.info("using cuda: {}".format(str(args.cuda)))
    device = "cuda:0" if args.cuda else "cpu"
    print("cuda available:", (device == "cuda:0"), ", using ", device)
    torch.set_num_threads(args.numcores)

    # specify data properties for caching
    name = "cached"
    name += "_min" + str(args.mincontig) + "_kmer" + str(args.kmer)
    if args.contignodes:
        name += "_contiggraph"
    dataset = ContigsDataset(
        name,
        args.assembly,
        assembly_name=args.assembly_name,
        graph_file=args.graph_file,
        save_dir=args.outdir,
        force_reload=args.reload,
        min_contig=args.mincontig,
        depth=args.depth,
        kmer=int(args.kmer),
        markers=args.markers,
        assembly_type=args.assembly_type,
        load_kmer=True,
        contig_nodes=args.contignodes
    )
    dataset.assembly = args.assembly

    if args.randomize:
        logger.info("generating a random graph")
        random_graph = dgl.rand_graph(len(dataset.node_names), len(dataset.edges_src))
        # random_graph = dgl.add_self_loop(random_graph)
        for k in dataset.graph.ndata:
            random_graph.ndata[k] = dataset.graph.ndata[k]
        random_graph.edata["weight"] = torch.ones(len(dataset.edges_src))
        dataset.graph = random_graph

    # filter graph by components
    dataset.connected = [c for c in dataset.connected if len(c) >= args.mincomp]

    #### select features to use
    # this code wont be used for now but next version will train AE from these features
    # zscore Kmer features (kmer are already loaded from reading the dataset)
    dataset.nodes_kmer = torch.FloatTensor(stats.zscore(dataset.nodes_kmer, axis=0))

    # Read depths from JGI file
    if args.depth is not None and os.path.isfile(os.path.join(args.assembly, args.depth)):
        dataset.depth = args.depth
        dataset.nodes_depths = []
        dataset.read_depths(os.path.join(args.assembly, args.depth))
        logging.debug("Abundance dim: {}".format(len(dataset.nodes_depths[0])))
        dataset.nodes_depths = torch.tensor(dataset.nodes_depths)
        if len(dataset.nodes_depths[0]) > 1:  # normalize depths
            dataset.nodes_depths = stats.zscore(dataset.nodes_depths, axis=0)
            depthssum = dataset.nodes_depths.sum(axis=1) + 1e-10
            dataset.nodes_depths /= depthssum.reshape((-1, 1))
    else:
        dataset.nodes_depths = torch.ones(dataset.nodes_kmer.shape[0], 1)

    ### prepare contig features with VAE
    batchsteps = []
    vamb_epochs = 500
    batchsteps = [25, 75, 150, 300]
    if len(dataset.nodes_depths[0]) == 1:
        vamb_bs = 32
    else:
        vamb_bs = 64
        
    nhiddens = [512, 512]
    print("using these batchsteps:", batchsteps)

    # features dir: if not set, use assembly dir if specified, else use outdir
    if args.features is None:
        if args.assembly != "":
            features_dir = os.path.join(args.assembly, "features.tsv")
        else:
            features_dir = os.path.join(args.outdir, "features.tsv")
    elif os.path.exists(os.path.join(args.assembly, args.features)):
        features_dir = os.path.join(args.assembly, args.features)
    elif os.path.exists(os.path.join(args.outdir, args.features)):
        features_dir = os.path.join(args.outdir, args.features)
    else:
        features_dir = args.features # user wants features to be created to a specific dir
    vamb_emb_exists = os.path.exists(features_dir)
    if args.vamb or not vamb_emb_exists:
        print("running VAMB...")
        vamb_outdir = os.path.join(args.outdir, "{}_vamb_out{}/".format(name, args.vambdim))
        vamb_logpath = os.path.join(vamb_outdir, "log.txt")
        if os.path.exists(vamb_outdir) and os.path.isdir(vamb_outdir):
            shutil.rmtree(vamb_outdir)
        os.mkdir(vamb_outdir)
        with open(vamb_logpath, "w") as vamb_logfile:
            run_vamb(
                outdir=vamb_outdir,
                fastapath=os.path.join(args.assembly, args.assembly_name),
                jgipath=os.path.join(args.assembly, args.depth),
                logfile=vamb_logfile,
                cuda=args.cuda,
                batchsteps=batchsteps,
                batchsize=vamb_bs,
                nepochs=vamb_epochs,
                mincontiglength=args.mincontig,
                nhiddens=nhiddens,
                nlatent=int(args.vambdim),
                norefcheck=True,
            )
            if args.assembly != "":
                shutil.copyfile(os.path.join(vamb_outdir, "embs.tsv"), features_dir)
            # args.features = "features.tsv"
            print("Contig features saved to {}".format(features_dir))

    # Read  features/embs from file in tsv format
    node_embs = {}
    print("loading features from", features_dir)
    with open(features_dir, "r") as ffile:
        for line in ffile:
            values = line.strip().split()
            node_embs[values[0]] = [float(x) for x in values[1:]]

    dataset.nodes_embs = [
        node_embs.get(n, np.random.uniform(10e-5, 1.0, len(values[1:]))) for n in dataset.node_names
    ]  # deal with missing embs
    dataset.nodes_embs = torch.FloatTensor(dataset.nodes_embs)

    # initialize empty features vector
    dataset.nodes_data = torch.FloatTensor(len(dataset.node_names), 0)
    if args.usekmer:
        dataset.nodes_data = torch.cat((dataset.nodes_data, dataset.nodes_kmer), dim=1)
    # if args.depth is not None:
    #    dataset.nodes_data = torch.cat((dataset.nodes_data, dataset.nodes_depths), dim=1)
    if args.features is not None:  # append embs
        dataset.nodes_data = torch.cat((dataset.nodes_data, dataset.nodes_embs), dim=1)
    dataset.graph.ndata["feat"] = dataset.nodes_data
    # dataset.graph.ndata["len"] = torch.Tensor(dataset.nodes_len)

    # Filter edges according to weight (could be from read overlap count or depth sim)
    if args.no_edges:
        dataset.filter_edges(10e6)
    # elif args.read_edges or args.depth:
    if args.edge_threshold is not None:
        dataset.filter_edges(int(args.edge_threshold))

    # All nodes have a self loop
    # dataset.graph = dgl.remove_self_loop(dataset.graph)
    # diff_edges = len(dataset.graph.edata["weight"])
    # dataset.graph = dgl.add_self_loop(dataset.graph)

    # max_weight = dataset.graph.edata["weight"].max().item()
    # dataset.graph.edata["weight"][diff_edges:] = max_weight
    dataset.graph.edata["weight"] = dataset.graph.edata["weight"].float()
    graph = dataset[0]
    logger.info(graph)
    graph = graph.to(device)

    # k can be user defined or dependent on the dataset
    k = len(dataset.species)
    if args.kclusters is not None:
        k = int(args.kclusters)

    # Load labels from file (eg binning results)
    if args.labels:
        unused_labels = 0
        total_labeled_nodes = 0
        logging.info("loading labels from {}".format(args.labels))
        node_to_label = {c: "NA" for c in dataset.contig_names}
        labels = set(["NA"])
        with open(args.labels, "r") as f:
            for line in f:
                # label, node = line.strip().split()
                if args.labels.endswith(".csv"):
                    values = line.strip().split(",")
                elif args.labels.endswith(".tsv"):  # amber format
                    if line.startswith("@"):
                        continue
                    values = line.strip().split("\t")
                node = values[0]
                label = values[1]
                total_labeled_nodes += 1
                if node in node_to_label:
                    node_to_label[node] = label
                    labels.add(label)
                else:
                    #print("unused label:", line.strip())
                    unused_labels += 1
        print(f"{unused_labels}/{total_labeled_nodes} labels not used")
        labels = list(labels)
        label_to_node = {s: [] for s in labels}
        for n in node_to_label:
            s = node_to_label[n]
            label_to_node[s].append(n)
        dataset.node_to_label = {n: l for n, l in node_to_label.items()}
        dataset.species = labels
        dataset.label_to_node = label_to_node
        # calculate homophily
        positive_edges = 0
        edges_without_label = 0
        for u, v in zip(dataset.edges_src, dataset.edges_dst):
            # breakpoint()
            if (
                dataset.contig_names[u] not in dataset.node_to_label
                or dataset.contig_names[v] not in dataset.node_to_label
            ):
                edges_without_label += 1
            if dataset.node_to_label[dataset.contig_names[u]] == dataset.node_to_label[dataset.contig_names[v]]:
                positive_edges += 1
        print(
            "homophily:",
            positive_edges / (len(dataset.graph.edges("eid")) - edges_without_label),
            len(dataset.graph.edges("eid")) - edges_without_label,
        )

    else:  # use dataset own labels (by default its only NA)
        label_to_node = {s: [] for s in dataset.species}
        node_to_label = {n: dataset.species[i] for n, i in dataset.node_to_label.items()}
        for n in dataset.node_to_label:
            # for s in dataset.node_to_label[n]:
            s = dataset.species[dataset.node_to_label[n]]
            label_to_node[s].append(n)

    # Load contig marker genes (Bacteria list)
    if args.markers is not None:
        logging.info("loading checkm results")
        ref_sets = read_marker_gene_sets(BACTERIA_MARKERS)
        contig_markers = read_contig_genes(os.path.join(args.assembly, args.markers))
        dataset.ref_marker_sets = ref_sets
        dataset.contig_markers = contig_markers
        marker_counts = get_markers_to_contigs(ref_sets, contig_markers)
        dataset.markers = marker_counts
    else:
        dataset.ref_marker_sets = None

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

    model = None
    if args.embs is None and args.read_embs is False:
        model = SAGE(
            graph.ndata["feat"].shape[1],
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
        if dataset.ref_marker_sets is not None and args.clusteringalgo is not None and not args.skip_preclustering:
            # cluster using only input features
            print("pre train clustering:")
            pre_cluster_to_contig, centroids = cluster_embs(
                dataset.graph.ndata["feat"].detach().cpu().numpy(),
                dataset.node_names,
                args.clusteringalgo,
                k,
                device=device,
                node_lens=np.array([c[0] for c in dataset.nodes_len]),
            )
            results = evaluate_contig_sets(dataset.ref_marker_sets, dataset.contig_markers, pre_cluster_to_contig)
            calculate_bin_metrics(results, logger=logger)

        if args.model == "sage":
            best_train_embs, best_model, last_train_embs, last_model = train_graphsage(
                dataset,
                model,
                batch_size=args.batchsize,
                fan_out=args.fanout,
                num_negs=args.negatives,
                neg_share=False,
                num_epochs=args.epoch,
                lr=args.lr,
                k=k,
                clusteringalgo=args.clusteringalgo,
                print_interval=args.print,
                loss_weights=(not args.no_loss_weights),
                sample_weights=(not args.no_sample_weights),
                logger=logger,
                device=device,
                epsilon=args.early_stopping,
                evalepochs=args.evalepochs,
            )

    else:
        if args.embs is not None:
            emb_file = args.embs
        else:
            emb_file = args.outdir + f"/{args.outname}_train_embs.pickle"
        with open(emb_file, "rb") as embsf:
            best_embs_dict = pickle.load(embsf)
            best_embs = np.array([best_embs_dict[i] for i in dataset.node_names])

    if "cluster" in args.post or "kmeans" in args.post:
        logger.info("clustering embs with {} ({})".format(args.clusteringalgo, k))
        # train_embs = last_train_embs

        if args.clusteringalgo is False:
            args.clusteringalgo = "kmeans"
        if model is None:
            best_train_embs = graph.ndata["feat"]
            last_train_embs = graph.ndata["feat"]
        if args.cuda:
            best_train_embs = best_train_embs.cpu()
            # last_train_embs should already be detached and on cpu
        best_cluster_to_contig, best_centroids = cluster_embs(
            best_train_embs.numpy(),
            dataset.node_names,
            args.clusteringalgo,
            # len(dataset.connected),
            k,
            device=device,
        )
        last_cluster_to_contig, last_centroids = cluster_embs(
            last_train_embs,
            dataset.node_names,
            args.clusteringalgo,
            # len(dataset.connected),
            k,
            device=device,
        )
        cluster_sizes = {}
        for c in best_cluster_to_contig:
            cluster_size = sum([len(dataset.contig_seqs[contig]) for contig in best_cluster_to_contig[c]])
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

        contig_lens = {dataset.contig_names[i]: dataset.nodes_len[i][0] for i in range(len(dataset.contig_names))}
        if len(dataset.species) > 1:
            evaluate_binning(best_cluster_to_contig, node_to_label, label_to_node, contig_sizes=contig_lens)
            # calculate overall P/R/F
            calculate_overall_prf(best_cluster_to_contig, best_contig_to_bin, node_to_label, label_to_node, contig_lens)
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
                contig_lens
            )
        if "writebins" in args.post:
            print("writing bins to ", args.outdir + "/{}_bins/".format(args.outname))
            # breakpoint()
            bin_dir = Path(args.outdir + "/{}_bins/".format(args.outname))
            bin_dir.mkdir(parents=True, exist_ok=True)
            [f.unlink() for f in bin_dir.glob("*.fa") if f.is_file()]
            clustered_contigs = set()
            multi_contig_clusters = 0
            print(len(best_cluster_to_contig), "clusters")
            short_contigs = set()
            skipped_clusters = 0
            for c in best_cluster_to_contig:
                cluster_size = sum([len(dataset.contig_seqs[contig]) for contig in best_cluster_to_contig[c]])
                if cluster_size < args.minbin:
                    # print("skipped small cluster", len(cluster_to_contig[c]), "contig")
                    for contig in best_cluster_to_contig[c]:
                        short_contigs.add(contig)
                    skipped_clusters += 1
                    continue
                multi_contig_clusters += 1
                with open(bin_dir / f"{c}.fa", "w") as binfile:
                    # breakpoint()
                    for contig in best_cluster_to_contig[c]:
                        binfile.write(">" + contig + "\n")
                        binfile.write(dataset.contig_seqs[contig] + "\n")
                        clustered_contigs.add(contig)
                # print("multi cluster", c, "size", cluster_size, "contigs", len(cluster_to_contig[c]))
            print("skipped {} clusters".format(skipped_clusters))
            single_clusters = multi_contig_clusters
            left_over = set(dataset.contig_names) - clustered_contigs - short_contigs
            for c in left_over:
                if c not in clustered_contigs and len(dataset.contig_seqs[c]) > args.minbin:

                    with open(bin_dir / f"{single_clusters}.fna", "w") as binfile:
                        binfile.write(">" + c + "\n")
                        binfile.write(dataset.contig_seqs[c] + "\n")
                        single_clusters += 1
                    # print("contig", single_clusters, "size", len(dataset.contig_seqs[c]))
            print("wrote", single_clusters, "clusters", multi_contig_clusters, ">= #contig", args.mincomp)
        if "contig2bin" in args.post:
            # invert cluster_to_contig
            logging.info("Writing contig2bin to {}/{}".format(args.outdir, args.outname))
            with open(args.outdir + f"/{args.outname}_best_contig2bin.tsv", "w") as f:
                f.write("@Version:0.9.0\n@SampleID:simHC+\n@@SEQUENCEID\tBINID\n")
                for c in best_contig_to_bin:
                    f.write(f"{str(c)}\t{str(best_contig_to_bin[c])}\n")
            last_contig_to_bin = {}
            for bin in last_cluster_to_contig:
                for contig in last_cluster_to_contig[bin]:
                    last_contig_to_bin[contig] = bin
            with open(args.outdir + f"/{args.outname}_last_contig2bin.tsv", "w") as f:
                f.write("@Version:0.9.0\n@SampleID:simHC+\n@@SEQUENCEID\tBINID\n")
                for c in last_contig_to_bin:
                    f.write(f"{str(c)}\t{str(last_contig_to_bin[c])}\n")

    # plot tsne embs
    if "tsne" in args.post:
        from sklearn.manifold import TSNE

        print("running tSNE")
        # filter only good clusters
        tsne = TSNE(n_components=2, random_state=SEED)
        if len(dataset.species) == 1:
            label_to_node = {c: cluster_to_contig[c] for c in hq_bins}
            label_to_node["mq/lq"] = []
            for c in cluster_to_contig:
                if c not in hq_bins:
                    label_to_node["mq/lq"] += list(cluster_to_contig[c])
        if centroids is not None:
            all_embs = tsne.fit_transform(torch.cat((torch.tensor(train_embs), torch.tensor(centroids)), dim=0))
            centroids_2dim = all_embs[train_embs.shape[0] :]
            node_embeddings_2dim = all_embs[: train_embs.shape[0]]
        else:
            centroids_2dim = None
            node_embeddings_2dim = tsne.fit_transform(torch.tensor(train_embs))
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
        # properties of all nodes
        nodeid_to_label = {i: node_to_label.get(n, "NA") for i, n in enumerate(dataset.node_names)}
        contig_lens = {i: dataset.nodes_len[i][0] for i in range(len(dataset.contig_names))}
        nodes_titles = {
            i: str(dataset.node_names[i]) + "<br>Length: " + str(contig_lens[i])
            for i in range(len(dataset.contig_names))
        }
        if dataset.depth is not None:
            nodes_titles = {
                i: nodes_titles[i] + "<br>Depth: " + ", ".join(["{:.4}".format(x) for x in dataset.nodes_depths[i]])
                for i in range(len(dataset.contig_names))
            }
        if cluster_to_contig:
            contig_to_cluster = {
                contig: cluster for cluster, contigs in cluster_to_contig.items() for contig in contigs
            }
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
                args.outdir + args.outname + "_" + str(i),
                contig_sizes=contig_lens,
                node_titles=nodes_titles,
            )
    if "edges" in args.post:
        print("writing edges to", args.outdir + args.outname + "_edges")
        with open(args.outdir + args.outname + "_edges", "w") as graphf:
            for e in zip(graph.edges()[0], graph.edges()[1]):
                graphf.write(str(e[0].item()) + "\t" + str(e[1].item()) + "\n")

    if "proximity" in args.post:
        breakpoint()
        dists = torch.bmm(train_embs, train_embs)
        edges_dist = dists * graph.adj()

    if "writeembs" in args.post:
        # write embs
        logger.info("writing best and last embs")
        best_train_embs = best_train_embs.cpu().detach().numpy()
        best_train_embs_dict = {dataset.node_names[i]: best_train_embs[i] for i in range(len(best_train_embs))}
        with open(os.path.join(args.outdir, f"{args.outname}_best_embs.pickle"), "wb") as f:
            pickle.dump(best_train_embs_dict, f)
        # last_train_embs = last_train_embs
        last_train_embs_dict = {dataset.node_names[i]: last_train_embs[i] for i in range(len(last_train_embs))}
        with open(os.path.join(args.outdir, f"{args.outname}_last_embs.pickle"), "wb") as f:
            pickle.dump(last_train_embs_dict, f)


if __name__ == "__main__":
    main()
