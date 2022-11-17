import argparse

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            contents = f.read()
        # parse arguments in the file and store them in a blank namespace
        data = parser.parse_args(contents.split(), namespace=None)
        for k, v in vars(data).items():
            # set arguments in the target namespace if they haven’t been set yet
            #if getattr(namespace, k, None) is None:
            #print(f"using args {k}={v}")
            setattr(namespace, k, v)

def create_parser():
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
    parser.add_argument("--model_name", type=str, help="One of the implemented models", default="sage_lstm")
    parser.add_argument(
        "--activation", type=str, help="Activation function to use(relu, prelu, sigmoid, tanh)", default="relu"
    )
    parser.add_argument("--layers_gnn", type=int, help="Number of layers of the GNN", default=3)
    parser.add_argument("--hidden_gnn", type=int, help="Dimension of hidden layers of GNN", default=128)
    parser.add_argument("--hidden_vae", type=int, help="Dimension of hidden layers of GNN", default=512)
    parser.add_argument("--embsize_gnn", "--zg", type=int, help="Output embedding dimension of GNN", default=32)
    parser.add_argument("--embsize_vae", "--zl", type=int, help="Output embedding dimension of VAE", default=64)
    parser.add_argument("--batchsize", type=int, help="batchsize to train the VAE", default=256)
    parser.add_argument("--batchtype", type=str, help="Batch type, nodes or edges", default="edges")
    parser.add_argument("--dropout_gnn", type=float, help="dropout of the GNN", default=0.0)
    parser.add_argument("--dropout_vae", type=float, help="dropout of the VAE", default=0.0)
    parser.add_argument("--lr_gnn", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--lr_vae", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--gnn_alpha", type=float, help="Coeficient for GNN loss", default=1)
    parser.add_argument("--kld_alpha", type=float, help="Coeficient for KLD loss", default=200)
    parser.add_argument("--ae_alpha", type=float, help="Coeficient for AE loss", default=1)
    parser.add_argument("--scg_alpha", type=float, help="Coeficient for SCG loss", default=1)
    parser.add_argument("--clusteringalgo", help="clustering algorithm", default="vamb")
    parser.add_argument("--kclusters", help="Number of clusters (only for some clustering methods)", default=None)
    # GraphSAGE params
    parser.add_argument("--aggtype", help="Aggregation type for GraphSAGE (mean, pool, lstm, gcn)", default="lstm")
    parser.add_argument("--decoder_input", help="What to use for input to the decoder", default="gnn")
    parser.add_argument("--ae_only", help="Do not use GNN (ae model must be used and decoder input must be ae", action="store_true")
    parser.add_argument("--negatives", help="Number of negatives to train GraphSAGE", default=10, type=int)
    parser.add_argument("--quick", help="Reduce number of nodes to run quicker", action="store_true")
    parser.add_argument("--classify", help="Run classification instead of clustering", action="store_true")
    parser.add_argument(
        "--fanout", help="Fan out, number of positive neighbors sampled at each level", default="10,25"
    )
    # other training params
    parser.add_argument("--epoch", type=int, help="Number of epochs to train model", default=100)
    parser.add_argument("--print", type=int, help="Print interval during training", default=10)
    parser.add_argument("--evalepochs", type=int, help="Epoch interval to run eval", default=20)
    parser.add_argument("--evalskip", type=int, help="Skip eval of these epochs", default=50)
    parser.add_argument("--eval_split", type=float, help="Percentage of dataset to use for eval", default=0.0)
    parser.add_argument("--kmer", default=4)
    parser.add_argument("--rawfeatures", help="Use raw features", action="store_true")
    parser.add_argument("--clusteringloss", help="Train with clustering loss", action="store_true")
    parser.add_argument(
        "--concat_features", help="Concat learned and original features before clustering", action="store_true"
    )
    parser.add_argument("--no_loss_weights", action="store_false", help="Using edge weights for loss (positive only)")
    parser.add_argument("--no_sample_weights", action="store_false", help="Using edge weights to sample negatives")
    parser.add_argument(
        "--early_stopping",
        type=float,
        help="Stop training if delta between last two losses is less than this",
        default="0.1",
    )
    parser.add_argument("--nruns", type=int, help="Number of runs", default=1)
    # data processing
    parser.add_argument("--mincontig", type=int, help="Minimum size of input contigs", default=1000)
    parser.add_argument("--minbin", type=int, help="Minimum size of clusters in bp", default=200000)
    parser.add_argument("--mincomp", type=int, help="Minimum size of connected components", default=1)
    parser.add_argument("--randomize", help="Randomize graph", action="store_true")
    parser.add_argument("--labelgraph", help="Create graph based on labels (ignore assembly graph)", action="store_true")
    parser.add_argument("--binarize", help="Binarize adj matrix", action="store_true")
    parser.add_argument("--noedges", help="Remove all but self edges from adj matrix", action="store_true")
    parser.add_argument("--read_embs", help="Read embeddings from file", action="store_true")
    parser.add_argument("--reload", help="Reload data", action="store_true")

    parser.add_argument("--markers", type=str, help="File with precomputed checkm results to eval",
                        default="marker_gene_stats.tsv")
    parser.add_argument("--post", help="Output options", default="writeembs_contig2bin")
    parser.add_argument("--skip_preclustering", help="Use precomputed checkm results to eval", action="store_true")
    parser.add_argument("--outname", "--outputname", help="Output (experiment) name", default="")
    parser.add_argument("--cuda", help="Use gpu", action="store_true")
    parser.add_argument("--noise", help="Use noise generator", action="store_true")
    parser.add_argument("--vamb", help="Run vamb instead of loading features file", action="store_true")
    parser.add_argument("--savemodel", help="Save best model to disk", action="store_true")
    parser.add_argument("--tsne", help="Plot tsne at checkpoints", action="store_true")
    parser.add_argument("--vambdim", help="VAE latent dim", default=32)
    parser.add_argument("--numcores", help="Number of cores to use", default=1, type=int)
    parser.add_argument(
        "--outdir", "--outputdir", help="Output dir (same as input assembly dir if not defined", default=None
    )
    parser.add_argument("--assembly_type", help="flye or spades", default="flye")
    parser.add_argument("--contignodes", help="Use contigs as nodes instead of edges", action="store_true")
    parser.add_argument("--seed", help="Set seed", default=1, type=int)
    parser.add_argument("--quiet", "-q", help="Do not output epoch progress", action="store_true")
    parser.add_argument("--read_cache", help="Do not check assembly files, read cached files only", action="store_true")
    parser.add_argument("--version", "-v", help="Print version and exit", action="store_true")
    parser.add_argument("--loglevel", "-l", help="Log level", default="info")
    parser.add_argument('--configfile', type=open, action=LoadFromFile)
    return parser
