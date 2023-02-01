# GraphMB: Assembly Graph Metagenomic Binner

# Introduction

GraphMB is a Metagenomic Binner developed for long-read assemblies, that takes advantage of graph machine learning 
algorithms and the assembly graph generated during assembly. It has been tested on (meta)flye assemblies.

*** NEW September 2022 *** 
Our peer-reviewed paper: 
> Andre Lamurias, Mantas Sereika, Mads Albertsen, Katja Hose, Thomas Dyhre Nielsen, Metagenomic binning with assembly graph embeddings, Bioinformatics, 2022;, btac557, https://doi.org/10.1093/bioinformatics/btac557

## Dependencies

GraphMB was developed with Python 3.7.
All of its dependencies are automatically installed.
For pre-processing and post-processing, check this README.

## Documentation

Can be found here: https://graphmb.readthedocs.io/en/latest/

## Installation

### NEW - Option 1 - From pypi
```bash
pip install graphmb
```

### Option 2 - From wheel
```bash
pip install https://github.com/AndreLamurias/GraphMB/releases/download/v0.1.5/graphmb-0.1.5-py3-none-any.whl
```

### Option 3 - From source
Clone this repository, and then:
```bash
cd GraphMB
python -m venv venv; source venv/bin/activate # optional
pip install .
```

### Option 4 - Docker
Either build the docker image with the Dockerfile or pull from dockerhub:
```bash
docker pull andrelamurias/graphmb
```

Then you can run GraphMB on a container. The image includes the Strong100 dataset. 
If you want to use other datasets, use the -v option to mount the path to your data.
```bash
docker run -it andrelamurias/graphmb bash
```


## Input files

The only files required are the contigs in fasta format, and the assembly graph in GFA format. For optimal performance,
the assembly graph should be generated with Flye 2.9, since it includes the number of reads mapping to each pair of
contigs. Also, for better results, CheckM is run on each contig using the general Bacteria marker sets. This is optional
though, you can just run the model for a number of epochs and pick the last model. The marker genes are also used to
improve model training through a constrained loss function.
By default, it runs with with early stopping.

In summary, you need to have a directory with these files (names can be changed with arguments):
- assembly.fasta: assembly graph edge sequences or contigs
- assembly_graph.gfa: assembly graph. it should have the sequences of assembly.fasta as nodes.
- assembly_depth.txt: output of `jgi_summarize_bam_contig_depths`
- marker_gene_stats.csv (optional): output of CheckM for each contig

If you do not specify this directory the `--assembly`, you have to use the full path of all files.
Otherwise, it will assume that the files are inside the directory.

You can get an example of these files [here](https://drive.google.com/drive/folders/1m6uTgTPUghk_q9GxfX1UNEOfn8jnIdt5?usp=sharing).
Download from this link and extract to data/strong100.
The datasets used in our experiments are available [here](https://zenodo.org/record/6122610)
These datasets include the VAE embeddings obtained with Vamb, which are automatically used by GraphMB.
If you want to re-run Vamb, use the `--vamb` option.
## How to run
If you have your assembly in some directory, with the files mentioned above:

```bash
graphmb --assembly data/strong100/ --outdir results/strong100/ --markers marker_gene_stats.tsv
```

The outdir will be created if it does not exist. The marker file is optional. 
You can specify the filenames:

```bash
graphmb --assembly data/strong100/ --outdir results/strong100/ --assembly_name edges.fasta --graph_file assembly_graph.gfa --depth edges_depth.txt --markers marker_gene_stats.tsv
```

By default GraphMB saves a TSV file mapping each contig to a bin to the assembly directory, as well as the weights and output embeddings of the best model.
The output directory can be changed with the `--outdir` argument.

To prevent GraphMB from run clustering after each epoch, do not provide the markers param or set the `--evalepochs` option to a higher number (default is 10)
This will make it run faster but the results might not be optimal.

```bash
graphmb --assembly data/strong100/ --outdir results/strong100/
```

To use GPU both for training and clustering, use the --cuda param:

```bash
graphmb --assembly data/strong100/ --outdir results/strong100/ --cuda
```


You can also run on CPU and limit the number of threads to use:
```bash
graphmb --assembly data/strong100/ --outdir results/strong100/ --numcores 4
```

GraphMB was tested on graphs where the nodes are contig paths and not full contigs.
This gives more granularity to the algorithm and the edges are directly obtained from the assembly graph.
However in some cases this might be inconvenient, for example if 
We implemented an option to use a graph where the nodes are contigs.
```bash
graphmb --assembly data/strong100/ --outdir results/strong100/ --assembly_name contigs.fasta --depth contig_depth.txt --contignodes
```

## Typical workflow
Our workflows are available [here](https://github.com/AndreLamurias/binning_workflows).
On this section we present an overview of how to get your data ready for GraphMB.

### Pre-processing

1. Assembly your reads with metaflye: ```flye -nano-raw <reads_file> -o <output> --meta```
2. Filter and polish assembly if necessary (or extract edge sequences and polish edge sequences instead)
```bash
mv assembly.fasta contigs.fasta
awk '/^S/{print ">"$2"\n"$3}' assembly_graph.gfa | fold > assembly.fasta
```
3. Convert assembly graph to contig-based graph if you want to use full contig instead of edges
4. Run CheckM on sequences with Bacteria markers: 
```bash
mkdir edges
cd edges; cat ../assembly.fasta | awk '{ if (substr($0, 1, 1)==">") {filename=(substr($0,2) ".fa")} print $0 > filename }'; cd ..
find edges/ -name "* *" -type f | rename 's/ /_/g'
# evaluate edges
checkm taxonomy_wf -x fa domain Bacteria edges/ checkm_edges/
checkm qa checkm_edges/Bacteria.ms checkm_edges/ -f checkm_edges_polished_results.txt --tab_table -o 2
```
5. Get abundances with `jgi_summarize_bam_contig_depths`:
```bash
minimap2 -I 64GB -d assembly.mmi assembly.fasta # make index
minimap2 -I 64GB -ax map-ont assembly.mmi <reads_file> > assembly.sam
samtools sort assembly.sam > assembly.bam
jgi_summarize_bam_contig_depths --outputDepth assembly_depth.txt assembly.bam
```
6. Now you should have all the files to run GraphMB: assembly.fasta, assembly_graph.gfa and assembly_depth.txt. The marker_gene_stats.csv file will be saved to checkm_edges/storage/.

We have only tested GraphMB on flye assemblies. Flye generates a repeat graph where the nodes do not correspond to full contigs. 
Depending on your setup, you need to either use the edges as contigs.

### Post-processing
To evaluate the binning output, we run checkM with the `--reduced-tree` option and count HQ bins (>90 completeness 
and <5 contamination).
The bins will be written to fasta files if `writebins` is included in the `--post` argument.
Alternatively, the `--post contig2bin` option can be used (on by default) and the bins from that file can be written 
to fasta with `src/write_fasta_bins.py`.
Then you can run checkM on the bins directory:
```bash
checkm lineage_wf -x fa --reduced_tree -t 4 --tab_table bins_dir/ outputdir/ -f outputdir/checkm.tsv
```
We also provide a script to filter the checkM output and get the number of HQ bins:
```bash
python src/process_checkm_results_expanded.py outputdir/checkm.tsv 90 5
```


## Parameters

GraphMB contains many parameters to run various experiments, from changing the data inputs, to architecture of the 
model, training loss, data preprocessing, and output type.
The defaults were chosen to obtain the published results on the WWTP datasets, but may require some tuning for different
datasets and scenarios.
The full list of parameters is below but this section focused on the most relevant ones.

### assembly, assembly_name, graph_file, features, labels, markers, depth
If --assembly is given, that is used as the base data directory. 
Every other path is in relation to that directory.
Otherwise, every data file path must be given in relation to the current directory

Note about markers: this file is not mandatory but assumed to exist by default.
This is because we have run all of our experiments with it.
Without this, the number of HQ bins will be probably worse.

### outdir, outname
Where to write the output files, including caches, and what prefix to use.
If not given, GraphMB writes to the data directory given by --assembly.

### reload

Ignore cache and reprocess data files.

### nruns, seed
Repeat experiment nrun times. Use --seed to specify the initial seed, which is changed
with every run to get different results.

### cuda

Run model training and clustering on GPU.

### contignodes

If the contigs given by the --assembly_name parameter are actual contigs and not assembly graph edges,
use this parameter, which will transform the assembly graph to use full contigs as nodes.

### model

Model to be used by GraphMB. By default it uses a Graph Convolution Network, and trains a Variational Autoencoder first
to generate node features. The VAE embeddings are saved to make it faster to rerun the GCN.

Other models: 
- sage_lstm: original GraphMB GraphSAGE model, requires DGL installation
- gat and sage: alternative GNN model to GCN (VAEGBin)
- gcn_ccvae, gat_ccvae, sage_ccvae: combined VAE and GNN models, trained end-to-end, or without GNN if layers_gnn=0

## Full list of parameters
```
usage: graphmb [-h] [--assembly ASSEMBLY] [--assembly_name ASSEMBLY_NAME] [--graph_file GRAPH_FILE] [--edge_threshold EDGE_THRESHOLD] [--depth DEPTH] [--features FEATURES] [--labels LABELS] [--embs EMBS] [--model_name MODEL_NAME]
               [--activation ACTIVATION] [--layers_vae LAYERS_VAE] [--layers_gnn LAYERS_GNN] [--hidden_gnn HIDDEN_GNN] [--hidden_vae HIDDEN_VAE] [--embsize_gnn EMBSIZE_GNN] [--embsize_vae EMBSIZE_VAE] [--batchsize BATCHSIZE]
               [--batchtype BATCHTYPE] [--dropout_gnn DROPOUT_GNN] [--dropout_vae DROPOUT_VAE] [--lr_gnn LR_GNN] [--lr_vae LR_VAE] [--graph_alpha GRAPH_ALPHA] [--kld_alpha KLD_ALPHA] [--ae_alpha AE_ALPHA] [--scg_alpha SCG_ALPHA]
               [--clusteringalgo CLUSTERINGALGO] [--kclusters KCLUSTERS] [--aggtype AGGTYPE] [--decoder_input DECODER_INPUT] [--vaepretrain VAEPRETRAIN] [--ae_only] [--negatives NEGATIVES] [--quick] [--classify] [--fanout FANOUT]
               [--epoch EPOCH] [--print PRINT] [--evalepochs EVALEPOCHS] [--evalskip EVALSKIP] [--eval_split EVAL_SPLIT] [--kmer KMER] [--rawfeatures] [--clusteringloss] [--targetmetric TARGETMETRIC] [--concatfeatures]
               [--no_loss_weights] [--no_sample_weights] [--early_stopping EARLY_STOPPING] [--nruns NRUNS] [--mincontig MINCONTIG] [--minbin MINBIN] [--mincomp MINCOMP] [--randomize] [--labelgraph] [--binarize] [--noedges]
               [--read_embs] [--reload] [--markers MARKERS] [--post POST] [--skip_preclustering] [--outname OUTNAME] [--cuda] [--noise] [--savemodel] [--tsne] [--numcores NUMCORES] [--outdir OUTDIR]
               [--assembly_type ASSEMBLY_TYPE] [--contignodes] [--seed SEED] [--quiet] [--read_cache] [--version] [--loglevel LOGLEVEL] [--configfile CONFIGFILE]

Train graph embedding model



options:
  -h, --help            show this help message and exit
  --assembly ASSEMBLY   Assembly base path
  --assembly_name ASSEMBLY_NAME
                        File name with contigs
  --graph_file GRAPH_FILE
                        File name with graph
  --edge_threshold EDGE_THRESHOLD
                        Remove edges with weight lower than this (keep only >=)
  --depth DEPTH         Depth file from jgi
  --features FEATURES   Features file mapping contig name to features
  --labels LABELS       File mapping contig to label
  --embs EMBS           No train, load embs
  --model_name MODEL_NAME
                        One of the implemented models
  --activation ACTIVATION
                        Activation function to use(relu, prelu, sigmoid, tanh)
  --layers_vae LAYERS_VAE
                        Number of layers of the VAE
  --layers_gnn LAYERS_GNN
                        Number of layers of the GNN
  --hidden_gnn HIDDEN_GNN
                        Dimension of hidden layers of GNN
  --hidden_vae HIDDEN_VAE
                        Dimension of hidden layers of VAE
  --embsize_gnn EMBSIZE_GNN, --zg EMBSIZE_GNN
                        Output embedding dimension of GNN
  --embsize_vae EMBSIZE_VAE, --zl EMBSIZE_VAE
                        Output embedding dimension of VAE
  --batchsize BATCHSIZE
                        batchsize to train the VAE
  --batchtype BATCHTYPE
                        Batch type, nodes or edges
  --dropout_gnn DROPOUT_GNN
                        dropout of the GNN
  --dropout_vae DROPOUT_VAE
                        dropout of the VAE
  --lr_gnn LR_GNN       learning rate
  --lr_vae LR_VAE       learning rate
  --graph_alpha GRAPH_ALPHA
                        Coeficient for graph loss
  --kld_alpha KLD_ALPHA
                        Coeficient for KLD loss
  --ae_alpha AE_ALPHA   Coeficient for AE loss
  --scg_alpha SCG_ALPHA
                        Coeficient for SCG loss
  --clusteringalgo CLUSTERINGALGO
                        clustering algorithm
  --kclusters KCLUSTERS
                        Number of clusters (only for some clustering methods)
  --aggtype AGGTYPE     Aggregation type for GraphSAGE (mean, pool, lstm, gcn)
  --decoder_input DECODER_INPUT
                        What to use for input to the decoder
  --vaepretrain VAEPRETRAIN
                        How many epochs to pretrain VAE
  --ae_only             Do not use GNN (ae model must be used and decoder input must be ae
  --negatives NEGATIVES
                        Number of negatives to train GraphSAGE
  --quick               Reduce number of nodes to run quicker
  --classify            Run classification instead of clustering
  --fanout FANOUT       Fan out, number of positive neighbors sampled at each level
  --epoch EPOCH         Number of epochs to train model
  --print PRINT         Print interval during training
  --evalepochs EVALEPOCHS
                        Epoch interval to run eval
  --evalskip EVALSKIP   Skip eval of these epochs
  --eval_split EVAL_SPLIT
                        Percentage of dataset to use for eval
  --kmer KMER
  --rawfeatures         Use raw features
  --clusteringloss      Train with clustering loss
  --targetmetric TARGETMETRIC
                        Metric to pick best epoch
  --concatfeatures      Concat learned and original features before clustering
  --no_loss_weights     Using edge weights for loss (positive only)
  --no_sample_weights   Using edge weights to sample negatives
  --early_stopping EARLY_STOPPING
                        Stop training if delta between last two losses is less than this
  --nruns NRUNS         Number of runs
  --mincontig MINCONTIG
                        Minimum size of input contigs
  --minbin MINBIN       Minimum size of clusters in bp
  --mincomp MINCOMP     Minimum size of connected components
  --randomize           Randomize graph
  --labelgraph          Create graph based on labels (ignore assembly graph)
  --binarize            Binarize adj matrix
  --noedges             Remove all but self edges from adj matrix
  --read_embs           Read embeddings from file
  --reload              Reload data
  --markers MARKERS     File with precomputed checkm results to eval
  --post POST           Output options
  --skip_preclustering  Use precomputed checkm results to eval
  --outname OUTNAME, --outputname OUTNAME
                        Output (experiment) name
  --cuda                Use gpu
  --noise               Use noise generator
  --savemodel           Save best model to disk
  --tsne                Plot tsne at checkpoints
  --numcores NUMCORES   Number of cores to use
  --outdir OUTDIR, --outputdir OUTDIR
                        Output dir (same as input assembly dir if not defined
  --assembly_type ASSEMBLY_TYPE
                        flye or spades
  --contignodes         Use contigs as nodes instead of edges
  --seed SEED           Set seed
  --quiet, -q           Do not output epoch progress
  --read_cache          Do not check assembly files, read cached files only
  --version, -v         Print version and exit
  --loglevel LOGLEVEL, -l LOGLEVEL
                        Log level

```
