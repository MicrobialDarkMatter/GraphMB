Introduction
============

GraphMB is a Metagenomic Binner developed for long-read assemblies, that takes advantage of graph machine learning 
algorithms and the assembly graph generated during assembly.
It has been tested on (meta)flye assemblies.

Installation
************

Option 1 - From wheel::
    pip install https://github.com/AndreLamurias/GraphMB/releases/download/v0.1.2/graphmb-0.1.2-py3-none-any.whl


Option 2 - From source::
    git clone https://github.com/AndreLamurias/GraphMB
    cd GraphMB
    python -m venv venv; source venv/bin/activate # optional
    pip install .


Option 3 - From anaconda::
    conda install -c andrelamurias graphmb


Input files
***********
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

You can get an example of these files https://drive.google.com/drive/folders/1m6uTgTPUghk_q9GxfX1UNEOfn8jnIdt5?usp=sharing
Download from this link and extract to data/strong100.

