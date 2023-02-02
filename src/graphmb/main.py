
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
import networkx as nx

from pathlib import Path
from graphmb.arg_options import create_parser
from graphmb.evaluate import (
    evaluate_contig_sets,
    calculate_overall_prf,
)
from graphmb.contigsdataset import AssemblyDataset
from graphmb.visualize import draw_nx_graph, run_tsne, plot_embs
from graphmb.utils import set_seed, get_cluster_mask
from graphmb.graphmb1 import (cluster_embs,
    evaluate_binning,
    calculate_bin_metrics,
    )

from graphmb.version import __version__

def run_model(dataset, args, logger, nrun, target_metric):
    if args.model_name.endswith("_ccvae"):
        from graphmb import train_ccvae
        return train_ccvae.run_model_ccvae(dataset, args, logger, nrun, target_metric)
    elif args.model_name == "vae":
        from graphmb import train_vae
        return train_vae.run_model_vae(dataset, args, logger, nrun)
    elif args.model_name in ("gcn", "sage", "gat"):
        from graphmb import train_gnn
        return train_gnn.run_model_gnn(dataset, args, logger, nrun, target_metric)

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
    # draw connected components to separate files
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
        print(f"Assembly Graph file {args.graph_file} not found, check --graph_file option")
    if not os.path.exists(os.path.join(args.assembly, args.features)) or not use_features:

        # needs assembly files to calculate features
        if not os.path.exists(os.path.join(args.assembly, args.assembly_name)):
            print(f"Assembly {args.assembly_name} not found, check --assembly_name option")
            exit()
        if not os.path.exists(os.path.join(args.assembly, args.depth)):
            print(f"Depth file {args.depth} not found, check --depth option, not using depths")
            #exit()


def get_activation(args):
    # pick activation function
    import torch.nn as nn
    if args.activation == "prelu":
        activation = nn.PReLU(args.hidden_gnn)
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
    from graphmb.graphsage_unsupervised import train_graphsage, SAGE
    activation = get_activation(args)
    model = SAGE(
        dataset.graph.ndata["feat"].shape[1],
        args.hidden_gnn,
        args.embsize_gnn,
        args.layers_gnn,
        activation,
        args.dropout_gnn,
        agg=args.aggtype
    )
    model = model.to(device)

    if model is not None:
        model = model.to(device)

    cluster_mask = get_cluster_mask(args.quick, dataset.assembly)
    logging.info(model)
    if (
        dataset.assembly.ref_marker_sets is not None
        and args.clusteringalgo is not None
        and not args.skip_preclustering
    ):
        # cluster using only input features
        logger.info("pre train clustering:")
        pre_cluster_to_contig, centroids = cluster_embs(
            dataset.graph.ndata["feat"].detach().cpu().numpy()[cluster_mask],
            np.array(dataset.assembly.node_names)[cluster_mask],
            args.clusteringalgo,
            args.kclusters,
            device=device,
            # node_lens=np.array([c[0] for c in dataset.assembly.nodes_len]),
            node_lens=np.array(dataset.assembly.node_lengths)[cluster_mask],
            seed=args.seed,
        )
        results = evaluate_contig_sets(
            dataset.assembly.ref_marker_sets, dataset.assembly.contig_markers, pre_cluster_to_contig
        )
        metrics = calculate_bin_metrics(results)
        logger.info(f"HQ: {len(metrics['hq'])}, MQ:, {len(metrics['mq'])} Total bins: {len(metrics['total'])}")

    best_train_embs, best_model, last_train_embs, last_model, metrics = train_graphsage(
        dataset,
        model,
        batch_size=args.batchsize,
        fan_out=args.fanout,
        num_negs=args.negatives,
        neg_share=False,
        num_epochs=args.epoch,
        lr=args.lr_gnn,
        k=args.kclusters,
        clusteringalgo=args.clusteringalgo,
        cluster_features=args.concatfeatures,
        print_interval=args.print,
        loss_weights=(not args.no_loss_weights),
        sample_weights=(not args.no_sample_weights),
        logger=logger,
        device=device,
        epsilon=args.early_stopping,
        evalepochs=args.evalepochs,
        seed=args.seed,
        eval_skip=args.evalskip,
        quick=args.quick
    )
    return best_train_embs, best_model, last_train_embs, last_model, metrics


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
    logger.info("### skipped {} clusters while writing to file".format(skipped_clusters))
    single_clusters = multi_contig_clusters
    left_over = set(dataset.node_names) - clustered_contigs - short_contigs
    for c in left_over:
        if c not in clustered_contigs and len(dataset.node_seqs[c]) > args.minbin:

            with open(bin_dir / f"{single_clusters}.fna", "w") as binfile:
                binfile.write(">" + c + "\n")
                binfile.write(dataset.node_seqs[c] + "\n")
                single_clusters += 1
            # print("contig", single_clusters, "size", len(dataset.contig_seqs[c]))
    logger.info(f"### wrote {single_clusters} clusters {multi_contig_clusters} >= #contig {args.mincomp}")


def run_post_processing(final_embs, args, logger, dataset, device, label_to_node, node_to_label, seed):
    metrics = {}
    if "cluster" in args.post or "kmeans" in args.post:
        logger.info("#### clustering embs with {} ({})".format(args.clusteringalgo, args.kclusters))
        # train_embs = last_train_embs

        if args.clusteringalgo is False:
            args.clusteringalgo = "kmeans"
        if not isinstance(final_embs, np.ndarray):
            if args.cuda:
                final_embs = final_embs.cpu()
            final_embs = final_embs.numpy()

            # last_train_embs should already be detached and on cpu
        best_cluster_to_contig, best_centroids = cluster_embs(
            final_embs,
            dataset.node_names,
            args.clusteringalgo,
            # len(dataset.connected),
            args.kclusters,
            device=device,
            seed=seed,
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
                    # logger.debug(
                    #    f"{binid}, {round(results[binid]['comp'],4)}, {round(results[binid]['cont'],4)}, "
                    #    f"{len(best_cluster_to_contig[binid])} {labels_count}"
                    # )
                    hq_bins.add(binid)
                    total_hq += 1
                if results[binid]["comp"] > 50 and results[binid]["cont"] < 10:
                    total_mq += 1
            logger.info("#### Total HQ {} ####".format(total_hq))
            logger.info("#### Total MQ {} ####".format(total_mq))
            metrics["hq_bins"] = total_hq
            metrics["mq_bins"] = total_mq
        contig_lens = {dataset.node_names[i]: dataset.node_lengths[i] for i in range(len(dataset.node_names))}
        if len(dataset.labels) > 1:
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
            logger.info(f"### writing bins to {args.outdir}/{args.outname}_bins/")
            write_bins(args, dataset, best_cluster_to_contig, logger)
        if "contig2bin" in args.post:
            # invert cluster_to_contig
            logger.info("### Writing contig2bin to {}/{}".format(args.outdir, args.outname))
            with open(args.outdir + f"/{args.outname}_best_contig2bin.tsv", "w") as f:
                f.write("@Version:0.9.0\n@SampleID:SAMPLEID\n@@SEQUENCEID\tBINID\n")
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
            outputname=os.path.join(args.outdir, args.outname + "_tsne_clusters.png"),
        )

        # node_sizes=[dataset.nodes_len[i][0] * 100 for i in range(len(dataset.contig_names))],
    if "draw" in args.post:
        print("drawing graph")
        draw(
            dataset,
            node_to_label,
            label_to_node,
            best_cluster_to_contig,
            os.path.join(args.outdir, args.outname + "_graph.png"),
            graph=graph,
        )

    if "edges" in args.post:
        logger.info(f"### writing edges to {args.outdir + args.outname}_edges")
        write_edges(graph, os.path.join(args.outdir, args.outname + "_edges"))

    if "writeembs" in args.post:
        logger.info("### writing best and last embs to {}".format(args.outdir))
        write_embs(final_embs, dataset.node_names, os.path.join(args.outdir, f"{args.outname}_best_embs.pickle"))
        # write_embs(best_train_embs, dataset.node_names, os.path.join(args.outdir, f"{args.outname}_last_embs.pickle"))
    return metrics

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.version:
        print(f"GraphMB {__version__}")
        exit(0)
    if not args.read_cache:
        check_dirs(args)
    # set up logging
    now = datetime.now()
    logger = logging.getLogger(__name__)
    loglevel = getattr(logging, args.loglevel.upper())
    logger.setLevel(loglevel)
    logfile = os.path.join(args.outdir, now.strftime("%Y%m%d-%H%M%S") + "{}_output.log".format(args.outname))
    output_file_handler = logging.FileHandler(logfile)
    print("logging to {}".format(logfile))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger("matplotlib").disabled = True
    logging.getLogger("matplotlib.pyplot").disabled = True

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    # setup tensorflow
    if args.model_name != "sage_lstm":
        if "torch" in sys.modules:
            sys.modules.pop('torch')
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
        import tensorflow as tf
        #tf.get_logger().setLevel(logging.INFO)
        clustering_device = "cpu" # avoid tf vs torch issues
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        if not args.cuda:
            tf.config.set_visible_devices([], "GPU")

    logger.info(f"Running GraphMB {__version__}")
    logger.debug(args)
    # setup cuda and cpu
    logger.info("using cuda: {}".format(str(args.cuda)))
    device = "cuda:0" if args.cuda else "cpu"
    clustering_device = "cuda:0" if args.cuda else "cpu"
    logger.info("setting seed to {}".format(args.seed))
    set_seed(args.seed)
    use_graph = os.path.exists(os.path.join(args.assembly, args.graph_file)) or args.read_cache
    # specify data properties for caching
    if args.features is None:
        if args.assembly != "":
            features_path = os.path.join(args.assembly, "features.tsv")
        else:
            features_path = os.path.join(args.outdir, "features.tsv")
    else:
        features_path = os.path.join(args.assembly, args.features)

    # create assembly object
    dataset = AssemblyDataset(
        name=args.outname,
        logger=logger,
        data_dir=args.assembly, #every file is appended to this this
        fastafile=args.assembly_name,
        graphfile=args.graph_file,
        depthfile=args.depth,
        scgfile=args.markers,
        labelsfile=args.labels,
        featuresfile=features_path,
        cache_dir=args.outdir,
        min_contig_length=args.mincontig,
        contignodes=args.contignodes
    )
    if args.read_cache or (dataset.check_cache(use_graph) and not args.reload):
        logger.info("Reading cache from".format(args.outdir))
        dataset.read_cache(use_graph)
    else:
        check_dirs(args, use_features=False)
        logger.info("Cache not found on {}".format(args.outdir))
        dataset.read_assembly()
        logger.info("")
    
    if args.markers.startswith("gtdb"):
        dataset.read_gtdbtk_files()
    elif args.markers != "":
        dataset.read_scgs()
    else:
        args.markers = None

    # load precomputed contigs with same SCGs (diff genomes)
    if os.path.exists(f"{dataset.cache_dir}/all_different.npy"):
        dataset.neg_pairs_idx = np.load(f"{dataset.cache_dir}/all_different.npy")
    elif args.markers is not None:
        dataset.get_all_different_idx()
        np.save(f"{dataset.cache_dir}/all_different.npy", dataset.neg_pairs_idx)
    else:
        dataset.neg_pairs_idx = np.array([])
        args.scg_alpha = 0
    
    if os.path.exists(os.path.join(args.assembly, "assembly_info.txt")):
        logger.info("Reading assembly info file")
        dataset.read_assembly_info()
        dataset.print_circular_contigs()
    # k can be user defined or dependent on the dataset
    if args.kclusters is None:
        args.kclusters = len(dataset.labels)
    args.kclusters = int(args.kclusters)

    # reload labels from file anyway
    if args.labels is not None:
        dataset.read_labels()
    
    if args.labelgraph:
        dataset.generate_edges_based_on_labels()
        dataset.calculate_homophily()
    
    dataset.print_stats()

    target_metric = "f1"
    if args.markers is not None:
        target_metric = "hq"
    elif args.labels is not None and "amber" in args.labels:
        target_metric = "f1_avg_bp"
    elif args.labels is None:
        target_metric = "noeval"

    # graph transformations
    # Filter edges according to weight (could be from read overlap count or depth sim)
    if not args.rawfeatures and args.model_name != "vae":
        if not os.path.exists(dataset.featuresfile):
            from graphmb import train_ccvae
            logger.info("==============Running VAE model=====================")
            old_args = copy.deepcopy(args)
            args.graph_alpha = 0 # do not use edges 
            args.outname = "ccvae"
            vae_embs, _ = train_ccvae.run_model_ccvae(dataset, args, logger, 0,
                                                      use_gnn=False, epochs=500,
                                                      target_metric=target_metric)
            logger.info("===================================================")
            dataset.node_embs = np.array(vae_embs)
            dataset.write_features_tsv()
            args = old_args
        else:
            dataset.read_features()
    
    # Prepare for running multiple runs and aggregate scores
    metrics_per_run = []
    amber_metrics_per_run = []
    for n in range(args.nruns):
        logger.info("RUN {}".format(n))
        if args.embs is not None:  # no training, just run post processing
            emb_file = args.embs
            with open(emb_file, "rb") as embsf:
                best_embs_dict = pickle.load(embsf)
                best_train_embs = np.array([best_embs_dict[i] for i in dataset.node_names])

        # DGL specific code - GraphMB1
        elif args.model_name == "sage_lstm":
            import torch
            from graphmb.dgl_dataset import DGLAssemblyDataset
            torch.set_num_threads(args.numcores)
            dgl_dataset = DGLAssemblyDataset(dataset)
            # initialize empty features vector
            nodes_data = torch.FloatTensor(len(dataset.node_names), 0)
            if args.rawfeatures:
                nodes_data = torch.cat(
                    (nodes_data, torch.FloatTensor(dataset.node_kmers), torch.FloatTensor(dataset.node_depths)), dim=1
                )
            # if args.depth is not None:
            #    dataset.nodes_data = torch.cat((dataset.nodes_data, dataset.nodes_depths), dim=1)
            elif args.features is not None:  # append embs
                node_embs = torch.FloatTensor(dataset.node_embs)
                nodes_data = torch.cat((nodes_data, node_embs), dim=1)
            dgl_dataset.graph.ndata["feat"] = nodes_data
            # dataset.graph.ndata["len"] = torch.Tensor(dataset.nodes_len)

            dgl_dataset.graph.edata["weight"] = dgl_dataset.graph.edata["weight"].float()
            graph = dgl_dataset[0]
            logger.info(graph)
            graph = graph.to(device)

            model = None
            if args.embs is None and args.read_embs is False:
                best_train_embs, model, last_train_embs, last_model, metrics = run_graphmb(dgl_dataset, args, device, logger)
                emb_file = args.outdir + f"/{args.outname}_train_embs.pickle"
                metrics = {k: len(v) for k, v in metrics.items()}
            if model is None:
                best_train_embs = graph.ndata["feat"]
                last_train_embs = graph.ndata["feat"]
        
        elif args.model_name in ("sage", "gcn", "gat", "vae", "vgae") or args.model_name.endswith("_ccvae") or \
             args.model_name.endswith("_decode") or args.model_name.endswith("_aug"):
            best_train_embs, metrics = run_model(dataset, args, logger, nrun=n, target_metric=target_metric)
            tf.keras.backend.clear_session()

        run_post_processing(
            best_train_embs,
            args,
            logger,
            dataset,
            clustering_device,
            dataset.label_to_node,
            dataset.node_to_label,
            seed=args.seed,
        )

        if args.labels is not None: # or "contig2bin" in args.post:
            from graphmb.amber_eval import amber_eval
            amber_metrics, bin_counts = amber_eval(
                os.path.join(args.assembly, args.labels), args.outdir + f"/{args.outname}_{n}_best_contig2bin.tsv", ["graphmb"]
            )
        #if args.labels is not None:
            hq = bin_counts["> 90% completeness"][1]
            mq = bin_counts["> 50% completeness"][1]
            amber_metrics["hq"] = hq
            amber_metrics["mq"] = mq
            amber_metrics_per_run.append(amber_metrics)
        metrics_per_run.append(metrics)
        args.seed += 1
        set_seed(args.seed)

    metrics_names = metrics_per_run[0].keys()
    for mname in metrics_names:
        values = [m.get(mname, 0) for m in metrics_per_run]
        logger.info("### {}: {:.3f} {:.3f}".format(mname, np.mean(values), np.std(values)))
    hqs = [m["hq"] for m in metrics_per_run]
    mqs = [m["mq"] for m in metrics_per_run]
    logger.info("{:.1f} {:.1f} {:.1f} {:.1f}".format(np.mean(hqs), np.std(hqs), np.mean(mqs), np.std(mqs)))
    if args.labels is not None:
        #amber_metrics_names = amber_metrics_per_run[0].keys()
        amber_metrics_names = ["precision_avg_bp", "recall_avg_bp", "f1_avg_bp", "hq", "mq"]
        for mname in amber_metrics_names:
            values = [m[mname] for m in amber_metrics_per_run]
            logger.info("### amber eval {}: {:.4f} {:.4f} ###".format(mname, np.mean(values), np.std(values)))
    total_time = datetime.now() - now
    print("Total run time: {}".format(total_time))
    print("Seconds per run: {:.2f}".format(total_time.total_seconds() / args.nruns))

if __name__ == "__main__":
    main()
