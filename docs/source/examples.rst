Examples
=========
If you have your assembly in some directory, with the files mentioned above:


.. code-block:: bash

    python src/graphmb/main.py  --assembly data/strong100/ --checkm_eval marker_gene_stats.tsv


You can specify the filenames:

.. code-block:: bash

    python src/graphmb/main.py  --assembly data/strong100/ --assembly_name edges.fasta --graph_file assembly_graph.gfa --depth edges_depth.txt --checkm_eval marker_gene_stats.tsv


To prevent GraphMB to run clustering after each epoch, do not provide the checkm_eval param.
This will make it run faster but the results might not be optimal.

.. code-block:: bash

    python src/graphmb/main.py  --assembly data/strong100/


To use GPU both for training and clustering, use the --cuda param:

.. code-block:: bash

    python src/graphmb/main.py  --assembly data/strong100/ --cuda



You can also run on CPU and limit the number of threads to use:
.. code-block:: bash

    python src/graphmb/main.py  --assembly data/strong100/ --numcores 4


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