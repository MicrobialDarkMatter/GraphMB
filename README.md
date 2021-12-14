# GraphMB: Assembly Graph Metagenomic Binner

# Introduction

GraphMB is a Metagenomic Binner developed for long-read assemblies, that takes advantage of graph machine learning 
algorithms and the assembly graph generated during assembly. It has been tested on (meta)flye assemblies.

## Dependencies

GraphMB was developed with Python 3.7, Pytorch and DGL.
It depends on VAMB to generate initial embeddings and  clustering, and CheckM to guide the training process and evaluate the output.
VAMB, Pytorch and DGL are installed automatically. 

## Installation

### Option 1 - From github
Clone this repository, and then:
```bash
cd GraphMB
python -m venv venv; source venv/bin/activate # optional
pip install .
```

### Option 2 - From anaconda
```bash
conda install -c andrelamurias graphmb
```

## Input files

The only files required are the contigs in fasta format, and the assembly graph in GFA format. For optimal performance,
the assembly graph should be generated with Flye 2.9, since it includes the number of reads mapping to each pair of
contigs. Also, for better results, CheckM is run on each contig using the general Bacteria marker sets. This is optional
though, you can just run the model for a number of epochs and pick the last model. 
By default, it runs with with early stopping.

In summary, you need to have a directory with these files:
- edges.fasta
- assembly_graph.fasta
- edges_depth.txt (output of `jgi_summarize_bam_contig_depths`)
- marker_gene_stats.csv (optional)

You can get an example of these files [here](https://drive.google.com/drive/folders/1m6uTgTPUghk_q9GxfX1UNEOfn8jnIdt5?usp=sharing).
Download from this link and extract to data/strong100.


## How to run
If you have your assembly in some directory, with the files mentioned above:

```bash
python src/graphmb/main.py  --assembly data/strong100/ --checkm_eval marker_gene_stats.tsv
```

You can specify the filenames:

```bash
python src/graphmb/main.py  --assembly data/strong100/ --assembly_name edges.fasta --graph_file assembly_graph.gfa --depth edges_depth.txt --checkm_eval marker_gene_stats.tsv
```

To prevent GraphMB to run clustering after each epoch, do not provide the checkm_eval param.
This will make it run faster but the results might not be optimal.

```bash
python src/graphmb/main.py  --assembly data/strong100/
```

To use GPU both for training and clustering, use the --cuda param:

```bash
python src/graphmb/main.py  --assembly data/strong100/ --cuda
```


You can also run on CPU and limit the number of threads to use:
```bash
python src/graphmb/main.py  --assembly data/strong100/ --numcores 4
```

If installed with pip, you can also use `graphmb` instead of `python src/graphmb/main.py`.

## Typical workflow
Our workflows are available [here](https://github.com/AndreLamurias/binning_workflows).
On this section we present an overview of how to get your data ready for GraphMB.

1. Assembly your reads with metaflye: ```flye -nano-raw <reads_file> -o <output> --meta```
2. Filter and polish assembly if necessary (or extract edge sequences and polish edge sequences instead)
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
jgi_summarize_bam_contig_depths --outputDepth asseembly_depth.txt assembly.bam
```
6. Now you should have all the files to run GraphMB

We have only tested GraphMB on flye assemblies. Flye generates a repeat graph where the nodes do not correspond to full contigs. 
Depending on your setup, you need to either use the edges as contigs.

TODO: postprocessing


## Full list of parameters
```bash
usage: main.py [-h] --assembly ASSEMBLY [--assembly_name ASSEMBLY_NAME] [--graph_file GRAPH_FILE] [--edge_threshold EDGE_THRESHOLD] [--depth DEPTH] [--features FEATURES] [--labels LABELS] [--markers MARKERS] [--embs EMBS] [--model MODEL] [--activation ACTIVATION]
               [--layers LAYERS] [--hidden HIDDEN] [--embsize EMBSIZE] [--batchsize BATCHSIZE] [--dropout DROPOUT] [--lr LR] [--clusteringalgo CLUSTERINGALGO] [--kclusters KCLUSTERS] [--aggtype AGGTYPE] [--negatives NEGATIVES] [--fanout FANOUT] [--epoch EPOCH] [--print PRINT]
               [--kmer KMER] [--usekmer] [--clusteringloss] [--no_loss_weights] [--no_sample_weights] [--early_stopping EARLY_STOPPING] [--mincontig MINCONTIG] [--minbin MINBIN] [--mincomp MINCOMP] [--randomize] [--no_edges] [--read_embs] [--reload] [--checkm_eval CHECKM_EVAL]
               [--post POST] [--skip_preclustering] [--outname OUTNAME] [--cuda] [--vamb] [--vambdim VAMBDIM] [--numcores NUMCORES]

Train graph embedding model

optional arguments:
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
  --markers MARKERS     File mapping nodes to SCG counts
  --embs EMBS           No train, load embs
  --model MODEL         only sage for now
  --activation ACTIVATION
                        Activation function to use(relu, prelu, sigmoid, tanh)
  --layers LAYERS       Number of layers of the GNN
  --hidden HIDDEN       Dimension of hidden layers of GNN
  --embsize EMBSIZE     Output embedding dimension of GNN
  --batchsize BATCHSIZE
                        batchsize to train the GNN
  --dropout DROPOUT     dropout of the GNN
  --lr LR               learning rate
  --clusteringalgo CLUSTERINGALGO
                        clustering algorithm
  --kclusters KCLUSTERS
                        Number of clusters (only for some clustering methods)
  --aggtype AGGTYPE     Aggregation type for GraphSAGE (mean, pool, lstm, gcn)
  --negatives NEGATIVES
                        Number of negatives to train GraphSAGE
  --fanout FANOUT       Fan out, number of positive neighbors sampled at each level
  --epoch EPOCH         Number of epochs to train model
  --print PRINT         Print interval during training
  --kmer KMER
  --usekmer             Use kmer features
  --clusteringloss      Train with clustering loss
  --no_loss_weights     Using edge weights for loss (positive only)
  --no_sample_weights   Using edge weights to sample negatives
  --early_stopping EARLY_STOPPING
                        Stop training if delta between last two losses is less than this
  --mincontig MINCONTIG
                        Minimum size of input contigs
  --minbin MINBIN       Minimum size of clusters in bp
  --mincomp MINCOMP     Minimum size of connected components
  --randomize           Randomize graph
  --no_edges            Add only self edges
  --read_embs           Read embeddings from file
  --reload              Reload data
  --checkm_eval CHECKM_EVAL
                        File with precomputed checkm results to eval
  --post POST           Output options
  --skip_preclustering  Use precomputed checkm results to eval
  --outname OUTNAME     Output (experiment) name
  --cuda                Use gpu
  --vamb                Run vamb instead of loading features file
  --vambdim VAMBDIM     VAE latent dim
  --numcores NUMCORES   Number of cores to use
```
