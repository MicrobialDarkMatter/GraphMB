Examples
=========
If you have your assembly in some directory, with the files mentioned above:


.. code-block:: bash
 
    graphmb  --assembly data/strong100/ --outdir results/strong100/

The outdir will be created if it does not exist.

You can specify the filenames:

.. code-block:: bash

   graphmb  --assembly data/strong100/ --outdir results/strong100/ --assembly_name edges.fasta --graph_file assembly_graph.gfa --depth edges_depth.txt --markers marker_gene_stats.tsv


.. code-block:: bash

    graphmb --assembly data/strong100/ --outdir results/strong100/ --assembly_name edges.fasta --graph_file assembly_graph.gfa --depth edges_depth.txt --markers marker_gene_stats.tsv


By default GraphMB saves a TSV file mapping each contig to a bin to the assembly directory, as well as the weights and output embeddings of the best model.
The output directory can be changed with the *--outdir* argument.

To prevent GraphMB from run clustering after each epoch, do not provide the markers param or set the `--evalepochs` option to a higher number (default is 10)
This will make it run faster but the results might not be optimal.

.. code-block:: bash

    graphmb --assembly data/strong100/ --outdir results/strong100/



To use GPU both for training and clustering, use the --cuda param:

.. code-block:: bash

    graphmb  --assembly data/strong100/ --cuda



You can also run on CPU and limit the number of threads to use:
.. code-block:: bash

    graphmb  --assembly data/strong100/ --numcores 4


If installed with pip, you can also use `graphmb` instead of `python src/graphmb/main.py`.

Typical workflow
****************
Our workflows are available [here](https://github.com/AndreLamurias/binning_workflows).
On this section we present an overview of how to get your data ready for GraphMB.

1. Assembly your reads with metaflye: ```flye -nano-raw <reads_file> -o <output> --meta```
2. Filter and polish assembly if necessary (or extract edge sequences and polish edge sequences instead)
3. Convert assembly graph to contig-based graph if you want to use full contig instead of edges
4. Run CheckM on sequences with Bacteria markers:: 

    mkdir edges
    cd edges; cat ../assembly.fasta | awk '{ if (substr($0, 1, 1)==">") {filename=(substr($0,2) ".fa")} print $0 > filename }'; cd ..
    find edges/ -name "* *" -type f | rename 's/ /_/g'
    # evaluate edges
    checkm taxonomy_wf -x fa domain Bacteria edges/ checkm_edges/
    checkm qa checkm_edges/Bacteria.ms checkm_edges/ -f checkm_edges_polished_results.txt --tab_table -o 2

5. Get abundances with `jgi_summarize_bam_contig_depths`::

    minimap2 -I 64GB -d assembly.mmi assembly.fasta # make index
    minimap2 -I 64GB -ax map-ont assembly.mmi <reads_file> > assembly.sam
    samtools sort assembly.sam > assembly.bam
    jgi_summarize_bam_contig_depths --outputDepth assembly_depth.txt assembly.bam
    
6. Now you should have all the files to run GraphMB

We have only tested GraphMB on flye assemblies. Flye generates a repeat graph where the nodes do not correspond to full contigs. 
Depending on your setup, you need to either use the edges as contigs.

Parameters
****************
GraphMB contains many parameters to run various experiments, from changing the data inputs, to architecture of the 
model, training loss, data preprocessing, and output type.
The defaults were chosen to obtain the published results on the WWTP datasets, but may require some tuning for different
datasets and scenarios.
The full list of parameters is below but this section focused on the most relevant ones.

assembly, assembly_name, graph_file, features, labels, markers, depth
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
If --assembly is given, that is used as the base data directory. 
Every other path is in relation to that directory.
Otherwise, every data file path must be given in relation to the current directory

Note about markers: this file is not mandatory but assumed to exist by default.
This is because we have run all of our experiments with it.
Without this, the number of HQ bins will be probably worse.

outdir, outname
""""""""""""""""""
Where to write the output files, including caches, and what prefix to use.
If not given, GraphMB writes to the data directory given by --assembly.

reload
""""""""""""""""""

Ignore cache and reprocess data files.

nruns, seed
""""""""""""""""""
Repeat experiment nrun times. Use --seed to specify the initial seed, which is changed
with every run to get different results.

cuda
""""""""""""""""""
Run model training and clustering on GPU.

contignodes
""""""""""""""""""
If the contigs given by the --assembly_name parameter are actual contigs and not assembly graph edges,
use this parameter, which will transform the assembly graph to use full contigs as nodes.

model
""""""""""""""""""
Model to be used by GraphMB. By default it uses a Graph Convolution Network, and trains a Variational Autoencoder first
to generate node features. The VAE embeddings are saved to make it faster to rerun the GCN.

Other models: 
- sage_lstm: original GraphMB GraphSAGE model, requires DGL installation
- gat and sage: alternative GNN model to GCN (VAEGBin)
- gcn_ccvae, gat_ccvae, sage_ccvae: combined VAE and GNN models, trained end-to-end, or without GNN if layers_gnn=0