from glob import escape
import itertools
import random
import os
import pickle
import pdb
import shutil
import torch
import dgl
import numpy as np
import networkx as nx
import scipy.stats as stats
import scipy.sparse
from dgl.data import DGLDataset
from dgl.data.utils import makedirs, save_info, load_info
from graphmb.graph_functions import read_reads_mapping_sam, count_kmers, get_kmer_to_id
from graphmb.evaluate import (
    read_marker_gene_sets,
    read_contig_genes,
    get_markers_to_contigs,
)
from vamb.vamb_run import run as run_vamb

BACTERIA_MARKERS = "data/Bacteria.ms"


def process_node_name(name, assembly_type):
    contig_name = name.strip().split(" ")[0]
    if assembly_type == "spades":
        contig_name = "_".join(contig_name.split("_")[:2])
    return contig_name


class AssemblyDataset:
    # Read contig names from fasta file and save to disk
    # read graph
    # other features are loaded by functions
    # everything as numpy to make it easier to convert
    def __init__(
        self,
        name,
        logger,
        data_dir,
        fastafile,
        graphfile,
        depthfile,
        scgfile,
        labelsfile,
        featuresfile,
        cache_dir,
        assemblytype="flye",
        min_contig_length=0,
    ):
        self.name = name
        self.logger = logger
        self.data_dir = data_dir
        self.fastafile = fastafile
        self.graphfile = graphfile
        self.depthfile = depthfile
        self.assembly_type = assemblytype
        self.cache_dir = cache_dir
        self.labelsfile = labelsfile
        self.featuresfile = featuresfile
        self.scgfile = scgfile
        self.load_kmer = True
        self.kmer = 4
        self.min_contig_length = min_contig_length
        if self.load_kmer:
            self.kmer_to_ids, self.canonical_k = get_kmer_to_id(self.kmer)

        # initialize data features which can be cached as numpy or pickle
        self.node_names = []
        self.node_seqs = {}  # should it be saved?
        self.graph_nodes = []  # nodes that are part of the assembly graph
        self.node_lengths = []
        self.node_kmers = []
        self.node_depths = []
        self.edges_src = []
        self.edges_dst = []
        self.edge_weights = []
        self.adj_matrix = None

        # initialize labels
        self.labels = []
        self.node_to_label = {}
        self.label_to_node = {}

    def read_assembly(self):
        """Read assembly files, convert to numpy arrays and save to disk"""
        self.logger.info("processing sequences {}".format(os.path.join(self.data_dir, self.fastafile)))
        self.read_seqs()
        self.logger.info(f"read {len(self.node_seqs)} seqs")
        np.save(os.path.join(self.cache_dir, "node_names.npy"), self.node_names)
        np.save(os.path.join(self.cache_dir, "node_lengths.npy"), self.node_lengths)
        np.save(os.path.join(self.cache_dir, "node_attributes_kmer.npy"), self.node_kmers.astype(float))
        pickle.dump(self.node_seqs, open(os.path.join(self.cache_dir, "node_seqs.pkl"), "wb"))
        self.logger.info("processing GFA file {}".format(os.path.join(self.data_dir, self.graphfile)))
        self.read_gfa()
        self.logger.info(f"read {len(self.edges_src)}, edges")
        np.save(os.path.join(self.cache_dir, "edge_weights.npy"), self.edge_weights)
        scipy.sparse.save_npz(os.path.join(self.cache_dir, "adj_sparse.npz"), self.adj_matrix)
        np.save(os.path.join(self.cache_dir, "graph_nodes.npy"), self.graph_nodes)
        # self.filter_contigs()
        # self.rename_nodes_to_index()
        # self.nodes_depths = np.array(self.nodes_depths)
        # read lengths
        self.logger.info("reading depths")
        self.read_depths()
        np.save(os.path.join(self.cache_dir, "node_attributes_depth.npy"), np.array(self.node_depths))
        self.logger.info("reading labels")
        self.read_labels()
        np.save(os.path.join(self.cache_dir, "node_to_label.npy"), self.node_to_label)
        np.save(os.path.join(self.cache_dir, "label_to_node.npy"), self.label_to_node)
        np.save(os.path.join(self.cache_dir, "labels.npy"), self.labels)
        # print("reading SCGs")
        # self.read_scgs()

    def read_cache(self):
        prefix = os.path.join(self.cache_dir, "{}")
        self.node_names = list(np.load(prefix.format("node_names.npy")))
        self.node_seqs = pickle.load(open(prefix.format("node_seqs.pkl"), "rb"))
        # graph nodes
        self.node_lengths = np.load(prefix.format("node_lengths.npy"))
        self.graph_nodes = np.load(prefix.format("graph_nodes.npy"))
        self.node_kmers = np.load(prefix.format("node_attributes_kmer.npy"))
        self.node_depths = np.load(prefix.format("node_attributes_depth.npy"))
        # self.edge_weights = np.load(prefix.format("edge_weights.npy"))
        self.adj_matrix = scipy.sparse.load_npz(prefix.format("adj_sparse.npz"))
        self.edge_weights = self.adj_matrix.data
        self.edges_src = self.adj_matrix.row
        self.edges_dst = self.adj_matrix.col
        # labels
        self.node_to_label = np.load("{}/node_to_label.npy".format(self.cache_dir), allow_pickle=True)[()]
        self.label_to_node = np.load("{}/label_to_node.npy".format(self.cache_dir), allow_pickle=True)[()]
        self.labels = list(np.load("{}/labels.npy".format(self.cache_dir)))

    def check_cache(self):
        """check if all necessary files exist in cache"""
        prefix = os.path.join(self.cache_dir, "{}")
        return (
            os.path.exists(prefix.format("node_names.npy"))
            and os.path.exists(prefix.format("node_lengths.npy"))
            and os.path.exists(prefix.format("node_attributes_kmer.npy"))
            and os.path.exists(prefix.format("node_attributes_depth.npy"))
            and os.path.exists(prefix.format("edge_weights.npy"))
            and os.path.exists(prefix.format("adj_sparse.npz"))
            and os.path.exists(prefix.format("graph_nodes.npy"))
            and os.path.exists(prefix.format("node_to_label.npy"))
            and os.path.exists(prefix.format("label_to_node.npy"))
            and os.path.exists(prefix.format("labels.npy"))
        )

    def read_seqs(self):
        """Read sequences from fasta file, write to self.node_seqs"""
        node_lengths = {}
        node_kmers = {}
        contig_name = None
        with open(os.path.join(self.data_dir, self.fastafile), "r") as f:
            for line in f:
                if line.startswith(">"):
                    if len(self.node_names) > 0:
                        # finish last contig
                        node_lengths[contig_name] = len(self.node_seqs[contig_name])
                        if self.load_kmer:
                            kmers = count_kmers(
                                self.node_seqs[contig_name], self.kmer, self.kmer_to_ids, self.canonical_k
                            )
                            node_kmers[contig_name] = kmers
                    contig_name = process_node_name(line[1:], self.assembly_type)
                    self.node_names.append(contig_name)
                    self.node_seqs[contig_name] = ""
                else:
                    self.node_seqs[contig_name] += line.strip()
        # add last
        node_lengths[contig_name] = len(self.node_seqs[contig_name])
        if self.load_kmer:
            kmers = count_kmers(self.node_seqs[contig_name], self.kmer, self.kmer_to_ids, self.canonical_k)
            node_kmers[contig_name] = kmers
            # convert kmers to numpy
            self.node_kmers = np.array(stats.zscore([node_kmers[n] for n in self.node_names], axis=0))
        self.node_lengths = [node_lengths[n] for n in self.node_names]

    def read_gfa(self):
        """Read graph file from GFA format, save list of start and end nodes to self.edges_src and self.edges_dst
        Check if contig names match the fasta file from self.node_seqs
        """
        skipped_contigs = set()
        with open(os.path.join(self.data_dir, self.graphfile), "r") as f:
            for line in f:
                if line.startswith("S"):  # sequence
                    values = line.strip().split()
                    node_name = process_node_name(values[1], self.assembly_type)
                    if node_name not in self.node_names:
                        skipped_contigs.add(node_name)
                        continue
                    contig_seq = self.node_seqs.get(node_name, "")  # discard missing contigs
                    contiglen = len(contig_seq)

                    if contiglen < self.min_contig_length:
                        skipped_contigs.add(node_name)
                    else:
                        self.graph_nodes.append(node_name)
                        # self.nodes_data.append([])
                if line.startswith("L"):  # link/edge
                    values = line.strip().split()  # TAG, SRC, SIGN, DEST, SIGN, 0M, RC
                    src_node_name = process_node_name(values[1], self.assembly_type)
                    dst_node_name = process_node_name(values[3], self.assembly_type)
                    if src_node_name in skipped_contigs or dst_node_name in skipped_contigs:
                        # skipped_edges.add((contig_names.index(values[1]), contig_names.index(values[3])))
                        continue
                    src_index = self.node_names.index(src_node_name)
                    dst_index = self.node_names.index(dst_node_name)
                    self.edges_src.append(src_index)
                    self.edges_dst.append(dst_index)
                    if len(values) > 6:
                        rc = int(values[6].split(":")[-1])
                    else:
                        rc = 1
                    self.edge_weights.append(rc)
                    # reverse too
                    if values[1] != values[3]:
                        self.edges_src.append(dst_index)
                        self.edges_dst.append(src_index)
                        self.edge_weights.append(rc)
            # TODO: Other information from gfa file: contig sequences
        self.logger.info(f"skipped contigs {len(skipped_contigs)} < {self.min_contig_length}")
        self.adj_matrix = scipy.sparse.coo_matrix(
            (self.edge_weights, (self.edges_src, self.edges_dst)), shape=(len(self.node_names), len(self.node_names))
        )

    def read_depths(self):
        node_depths = {}
        if self.depthfile is not None:
            with open(os.path.join(self.data_dir, self.depthfile)) as f:
                header = next(f)
                depth_i = [i + 3 for i, n in enumerate(header.split("\t")[3:]) if "-var" not in n]
                # var_i = [i + 3 for i, n in enumerate(header.split("\t")[3:]) if "-var" in n]
                for line in f:
                    values = line.strip().split()
                    node_name = process_node_name(values[0], self.assembly_type)
                    # if self.assembly_type == "spades":
                    #    node_name = "_".join(node_name.split("_")[:2])
                    if node_name in self.node_names:
                        node_depths[node_name] = np.array([float(values[i]) for i in depth_i])
                    else:
                        self.logger.info("node name not found: {}".format(node_name))
            self.node_depths = np.array([node_depths[n] for n in self.node_names])
            if len(self.node_depths[0]) > 1:  # normalize depths
                self.node_depths = stats.zscore(self.node_depths, axis=0)
                depthssum = self.node_depths.sum(axis=1) + 1e-10
                self.node_depths /= depthssum.reshape((-1, 1))
        else:
            self.node_depths = np.ones(len(self.node_names), 1)

    def read_labels(self):
        # logging.info("loading labels from {}".format(args.labels))
        if self.labelsfile is not None:
            node_to_label = {c: "NA" for c in self.node_names}
            labels = set(["NA"])
            with open(os.path.join(self.data_dir, self.labelsfile), "r") as f:
                for line in f:
                    # label, node = line.strip().split()
                    if self.labelsfile.endswith(".csv"):
                        values = line.strip().split(",")
                    elif self.labelsfile.endswith(".tsv"):  # amber format
                        if line.startswith("@"):
                            continue
                        values = line.strip().split("\t")
                    node = values[0]
                    label = values[1]
                    if node in node_to_label:
                        node_to_label[node] = label
                        labels.add(label)
                    else:
                        self.logger.info(("unused label: {}".format(line.strip())))
            labels = list(labels)
            label_to_node = {s: [] for s in labels}
            for n in node_to_label:
                s = node_to_label[n]
                label_to_node[s].append(n)
            self.node_to_label = {n: l for n, l in node_to_label.items()}
            self.labels = labels
            self.label_to_node = label_to_node
            # calculate homophily
            self.calculate_homophily()
        else:
            self.labels = ["NA"]
            self.label_to_node = {"NA": self.node_names}
            self.node_to_label = {n: "NA" for n in self.node_names}

    def calculate_homophily(self):
        positive_edges = 0
        edges_without_label = 0
        for u, v in zip(self.edges_src, self.edges_dst):
            # breakpoint()
            if self.node_names[u] not in self.node_to_label or self.node_names[v] not in self.node_to_label:
                edges_without_label += 1
            if self.node_to_label[self.node_names[u]] == self.node_to_label[self.node_names[v]]:
                positive_edges += 1
        self.logger.info(
            f"homophily: {positive_edges / (len(self.edge_weights) - edges_without_label)} {len(self.edge_weights) - edges_without_label}"
        )

    def read_scgs(self):
        # Load contig marker genes (Bacteria list)
        if self.scgfile is not None:
            # logging.info("loading checkm results")
            ref_sets = read_marker_gene_sets(BACTERIA_MARKERS)
            contig_markers = read_contig_genes(self.scgfile)
            self.ref_marker_sets = ref_sets
            self.contig_markers = contig_markers
            marker_counts = get_markers_to_contigs(ref_sets, contig_markers)
            self.markers = marker_counts
        else:
            self.ref_marker_sets = None

    def run_vamb(self, vamb_outdir, cuda, vambdim):
        self.logger.info("running VAMB")
        batchsteps = []
        vamb_epochs = 500
        if len(self.node_depths[0]) == 1:
            vamb_bs = 32
            batchsteps = [25, 75, 150]
        else:
            vamb_bs = 64
            batchsteps = [25, 75, 150, 300]
        nhiddens = [512, 512]
        self.logger.info("using these batchsteps:{}".format(batchsteps))

        vamb_logpath = os.path.join(vamb_outdir, "log.txt")
        if os.path.exists(vamb_outdir) and os.path.isdir(vamb_outdir):
            shutil.rmtree(vamb_outdir)
        os.mkdir(vamb_outdir)
        with open(vamb_logpath, "w") as vamb_logfile:
            run_vamb(
                outdir=vamb_outdir,
                fastapath=os.path.join(self.data_dir, self.fastafile),
                jgipath=os.path.join(self.data_dir, self.depthfile),
                logfile=vamb_logfile,
                cuda=cuda,
                batchsteps=batchsteps,
                batchsize=vamb_bs,
                nepochs=vamb_epochs,
                mincontiglength=self.min_contig_length,
                nhiddens=nhiddens,
                nlatent=int(vambdim),
                norefcheck=True,
            )
            if self.data_dir != "":
                shutil.copyfile(os.path.join(vamb_outdir, "embs.tsv"), self.featuresfile)
            # args.features = "features.tsv"
            self.logger.info("Contig features saved to {}".format(self.featuresfile))

    def read_features(self):
        node_embs = {}
        self.logger.info("loading features from {}".format(self.featuresfile))
        with open(self.featuresfile, "r") as ffile:
            for line in ffile:
                values = line.strip().split()
                node_embs[values[0]] = [float(x) for x in values[1:]]
        self.logger.info("loaded {} features/ {} nodes".format(len(node_embs), len(self.node_names)))
        self.node_embs = [
            node_embs.get(n, np.random.uniform(10e-5, 1.0, len(values[1:]))) for n in self.node_names
        ]  # deal with missing embs
        self.node_embs = np.array(self.node_embs)


class DGLAssemblyDataset(DGLDataset):
    def __init__(self, assembly):
        self.assembly = assembly
        self.logger = assembly.logger
        super().__init__(name=assembly.name + "assembly_graph", save_dir=assembly.cache_dir, force_reload=False)

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    def process(self, root=False):
        """Open GFA file to DGL format"""
        # TODO: this should receive and assembly dataset object and initialize self.graph
        # if root:
        #    root_node = G.add_node("root", length=0)
        # TODO: skip unconnected and too short/big
        self.logger.info("creating DGL graph")
        self.graph = dgl.graph(
            (self.assembly.edges_src, self.assembly.edges_dst),
            num_nodes=len(self.assembly.node_names),
        )
        self.graph.edata["weight"] = torch.tensor(self.assembly.edge_weights)

        self.logger.info("done")
        self.graph.ndata["label"] = torch.LongTensor(
            [self.assembly.labels.index(self.assembly.node_to_label[n]) for n in self.assembly.node_names]
        )

        nx_graph = self.graph.to_networkx().to_undirected()
        self.logger.info("connected components...")
        # self.connected = [c for c in sorted(nx.connected_components(nx_graph), key=len, reverse=True) if len(c) > 1]
        # breakpoint()
        self.connected = [c for c in sorted(nx.connected_components(nx_graph), key=len, reverse=True) if len(c) > 0]
        self.logger.info((len(self.connected), "connected"))
        # for group in self.connected:
        #    self.graphs.append(
        #        dgl.node_subgraph(self.graph, [self.node_names.index(c) for c in group if c in self.node_names])
        #    )

        assert len([c for comp in self.connected for c in comp]) <= len(self.assembly.node_names)

        # self.set_node_mask()

    def save(self):
        pass
        # save graphs and labels
        # save other information in python dict
        """info_path = os.path.join(self.save_path, "cache.pkl")
        print("saving graph", info_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        save_info(info_path, vars(self))"""

    def load(self):
        pass
        # load processed data from directory `self.save_path`
        """ info_path = os.path.join(self.save_path, "cache.pkl")
        print("loading from", info_path)
        loaded_info = load_info(info_path)
        for key in loaded_info:
            setattr(self, key, loaded_info[key])"""

    def has_cache(self):
        """# check whether there are processed data in `self.save_path`
        info_path = os.path.join(self.save_path, "cache.pkl")
        return os.path.exists(info_path)"""
        return False


class ContigsDataset(DGLDataset):
    def __init__(
        self,
        name,
        assembly_path=None,
        assembly_name="assembly.fasta",
        graph_file="assembly_graph.gfa",
        labels=None,
        save_dir=None,
        force_reload=False,
        min_contig=1000,
        kmer=4,
        depth=None,
        markers=None,
        load_kmer=False,
        assembly_type="flye",
    ):
        self.mode = "train"
        # self.save_dir = save_dir
        self.assembly = assembly_path
        if self.assembly is None:
            self.assembly = ""
        self.readmapping = assembly_path
        self.assembly_name = assembly_name
        self.graph_file = graph_file
        self.depth = depth
        self.markers = markers
        self.contig_names = []
        self.contig_seqs = {}
        self.read_names = []
        self.node_names = []  # contig_names + read_names
        self.nodes_len = []
        self.nodes_depths = []
        self.nodes_markers = []
        self.nodes_kmer = []
        self.graphs = []
        self.edges_src = []
        self.edges_dst = []
        self.edges_weight = []
        self.nodes_data = []
        self.node_to_label = {}
        self.node_labels = []
        self.kmer = kmer
        self.load_kmer = load_kmer
        self.assembly_type = assembly_type
        if self.load_kmer:
            self.kmer_to_ids, self.canonical_k = get_kmer_to_id(self.kmer)

        self.connected = []
        self.min_contig_len = min_contig
        if labels is None:
            self.species = ["NA"]
            self.add_new_species = True
        else:
            self.species = labels
            self.add_new_species = False
        super().__init__(name=name, save_dir=save_dir, force_reload=force_reload)

    def filter_edges(self, weight=0):
        """Filter edges based on weight"""
        # print(max(self.edges_weight), min(self.edges_weight))
        if weight < 0:
            weight = sum(self.edges_weight) / len(self.edges_weight)
        # for i in range(max(self.edges_weight)):
        #    print(i, self.edges_weight.count(i))
        keep_idx = self.graph.edata["weight"] >= weight
        idx_to_remove = [i for i, x in enumerate(keep_idx) if not x]
        self.edges_src = [self.edges_src[i] for i in keep_idx]
        self.edges_dst = [self.edges_dst[i] for i in keep_idx]
        # self.graph.edata["weight"] = self.graph.edata["weight"][keep_idx]
        self.graph.remove_edges(idx_to_remove)

    def filter_contigs(self):
        # remove disconnected
        keep_idx = [i for i, c in enumerate(self.contig_names) if c in self.edges_src or c in self.edges_dst]
        # keep_idx = [i for i, c in enumerate(self.contig_names)]
        self.contig_names = [self.contig_names[i] for i in keep_idx]
        self.nodes_kmer = [self.nodes_kmer[i] for i in keep_idx]
        self.nodes_depths = [self.nodes_depths[i] for i in keep_idx]
        self.nodes_len = [self.nodes_len[i] for i in keep_idx]

    def remove_nodes(self, remove_list):
        self.graph.remove_nodes(torch.tensor(remove_list))
        # self.contig_seqs = {}
        self.node_names = [self.node_names[i] for i in range(len(self.node_names)) if i not in remove_list]
        self.nodes_len = [self.nodes_len[i] for i in range(len(self.nodes_len)) if i not in remove_list]
        self.nodes_depths = [self.nodes_depths[i] for i in range(len(self.nodes_depths)) if i not in remove_list]
        self.nodes_markers = [self.nodes_markers[i] for i in range(len(self.nodes_markers)) if i not in remove_list]
        self.nodes_kmer = [self.nodes_kmer[i] for i in range(len(self.nodes_kmer)) if i not in remove_list]
        # self.edges_src = []
        # self.edges_dst = []
        # self.edges_weight = []
        self.nodes_data = [self.nodes_data[i] for i in range(len(self.nodes_data)) if i not in remove_list]
        # self.node_to_label = {}
        self.node_labels = [self.node_labels[i] for i in range(len(self.node_labels)) if i not in remove_list]

    def rename_nodes_to_index(self):
        # self.edges_src = [self.contig_names.index(i) for i in self.edges_src if i in self.contig_names]
        # self.edges_dst = [self.contig_names.index(i) for i in self.edges_dst if i in self.contig_names]
        edge_name_to_index = {n: i for i, n in enumerate(self.contig_names)}
        self.edges_src = [edge_name_to_index[n] for n in self.edges_src]
        self.edges_dst = [edge_name_to_index[n] for n in self.edges_dst]

    def set_node_mask(self):
        """Set contig nodes"""
        self.graph.ndata["contigs"] = torch.zeros(len(self.node_names), dtype=torch.bool)
        self.graph.ndata["contigs"][: len(self.contig_names)] = True

    def get_labels_from_reads(self, reads, add_new=True):
        contig_to_species = {}
        read_to_species = {}
        # contig_lens = {}
        for r in reads:
            # print(r)
            speciesname = r.split("_reads")[0]

            if speciesname not in self.species:
                if add_new:
                    self.species.append(speciesname)
                else:
                    continue
            for m in reads[r].mappings:
                if m.contigname not in contig_to_species:  # and m.contigname in contig_lens:
                    contig_to_species[m.contigname] = {}
                if speciesname not in contig_to_species[m.contigname]:
                    contig_to_species[m.contigname][speciesname] = 0
                # contig_to_species[m.contigname][speciesname] += m.mapq  # weight mapping by quality
                contig_to_species[m.contigname][speciesname] += 1  # weight mapping by len
                read_to_species[r] = speciesname
                # reads_count = int(values[0])
            # contig_to_species[values[1]][speciesname] = reads_count
        return contig_to_species, read_to_species
