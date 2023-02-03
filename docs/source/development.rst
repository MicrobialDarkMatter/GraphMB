Development
===========

Code structure
****************

GraphMB contains options to experiment with model architecture and training, 
as well as with pre-processing of data and post-processing of the results.
The core of GraphMB are deep learning models that process contigs into an 
embedding space.

The files **models.py**, **gnn_models.py**, and **layers.py** contain the 
tensorflow models used on version 0.2. 
These fails also contain the trainer helper function and loss functions.
The files **train_ccvae.py** and **train_gnn.py** contain the training loops of those models.
**graphsage_unsupervised.py** contains the model used by v0.1, while **graphmb1.py**
contains helper functions used on the initial GraphMB release (but not anymore).

The file **evaluate.py** contains several evaluation metrics, and a function to
run a clustering algorithm on the embeddings and evaluate the output.
The main clustering algorithm is in **vamb_clustering.py** as was originally developed
for VAMB. The file **amber_eval.py** is adapted from the AMBER evaluation tool, to run
the same metrics as that tool. 

The file **contigsdataset.py** contains the code to read, pre-process and save a
set of contigs, along with their assembly graph, depth, single-copy marker genes 
and embeddings. It also computes several stats on a dataset.
The file **dgl_dataset.py** contains code to convert the v0.2 AssemblyDataset class
to the one used by v0.1.
The file **utils.py** contains some additional helper functions that did not fit 
elsewhere. 

Finally, all the running parameters are stored in **arg_options.py**.
The main file **main.py** reads these parameters and executes the experiments
accordingly. **version.py** is used only to store the current version number.
**setup.py** defines the dependencies and other parameters to build a new version.



Typical workflow for new versions
**********************************
Useful commands to build new version:

.. code-block:: bash

    python setup.py sdist bdist_wheel
    python -m twine upload  dist/graphmb-X.X.X*
    cd docs; make html
    sudo docker build . -t andrelamurias/graphmb:X.X.X
    sudo docker push


Documentation
****************
The documentation is stored in docs/ and uses Sphinx to generate HTML pages.
The docstring of each funtion and class are automatically added. If new source 
code files are added, these should be added too to docs/source/graphmb.rst.
