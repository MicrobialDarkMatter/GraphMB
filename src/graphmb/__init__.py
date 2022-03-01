import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
from . import contigsdataset
from . import graph_functions
from . import evaluate
from . import graphsage_unsupervised
from . import version
