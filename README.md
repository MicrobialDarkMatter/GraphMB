# GraphEmb

# Introduction

GraphEmb is a metagenomics binner developed for long-read assemblies, that takes advantage of graph machine learning 
algorithms and the assembly graph generated during assembly. It has been tested on (meta)flye assemblies.

## Dependencies

GraphEmb was developed with Python 3.7, Pytorch and DGL.
It depends on VAMB to generate initial embeddings and  clustering, and CheckM to guide the training process and evaluate the output.
VAMB, Pytorch and DGL are installed automatically. 

## Installation

### Option 1 - From github
Clone this repository, and then:
```bash
cd GraphMB
python -m venv venv; source venv/bin/activate # optional
pip install -e
```


## Input files

The only files required are the contigs in fasta format, and the assembly graph in GFA format. For optimal performance,
the assembly graph should be generated with Flye 2.9, since it includes the number of reads mapping to each pair of
contigs. Also, for better results, CheckM is run on each contig using the general Bacteria marker sets. This is optional
though, you can just run the model for a number of epochs and pick the last model. 
By default, it runs with with early stopping.

In summary, you need:
- assembly.fasta
- assembly_graph.fasta
- output of `jgi_summarize_bam_contig_depths`
- marker_gene_stats.csv (optional)

You can get an example of these files [here](https://drive.google.com/drive/folders/1m6uTgTPUghk_q9GxfX1UNEOfn8jnIdt5?usp=sharing).
Download from this link and extract to data/strong100.


## How to run
For example:
```bash
python src/graphmb/main.py  --assembly data/strong100/ --assembly_name edges.fasta --graph_file assembly_graph.gfa --depth edges_depth.txt --checkm_eval marker_gene_stats.tsv
```
This assumes the existence of a edges.fasta, assembly_graph.gfa and marker_gene_stats.tsv and edges_depth.txt files on ./data/strong100.

TODO: preprocessing flye assembly

We have only tested GraphEmb on flye assemblies. Flye generates a repeat graph where the nodes do not correspond to full contigs. 
Depending on your setup, you need to either use the edges as contigs.
