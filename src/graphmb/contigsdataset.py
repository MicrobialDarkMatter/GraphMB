from glob import escape
import itertools
import random
import os
import pickle
import pdb
import shutil

import numpy as np
import networkx as nx
import scipy.stats as stats
import scipy.sparse
from graphmb.graph_functions import read_reads_mapping_sam, count_kmers, get_kmer_to_id
from graphmb.evaluate import (
    read_marker_gene_sets,
    read_contig_genes,
    get_markers_to_contigs,
)


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
        self.graph_paths = {} # contig paths identified by assembler
        self.node_lengths = []
        self.node_kmers = []
        self.node_depths = []
        self.edges_src = []
        self.edges_dst = []
        self.edge_weights = []
        self.adj_matrix = None
        self.neg_pairs_idx = None  # cached on the main function, only necessary for TF models

        # initialize labels
        self.labels = []
        self.node_to_label = {}
        self.label_to_node = {}
        
        self.contig_markers = {}

    def read_assembly(self):
        """Read assembly files, convert to numpy arrays and save to disk"""
        self.logger.info("processing sequences {}".format(os.path.join(self.data_dir, self.fastafile)))
        self.read_seqs()
        self.logger.info(f"read {len(self.node_seqs)} seqs")
        np.save(os.path.join(self.cache_dir, "node_names.npy"), self.node_names)
        np.save(os.path.join(self.cache_dir, "node_lengths.npy"), self.node_lengths)
        np.save(os.path.join(self.cache_dir, "node_attributes_kmer.npy"), self.node_kmers.astype(float))
        pickle.dump(self.node_seqs, open(os.path.join(self.cache_dir, "node_seqs.pkl"), "wb"))
        if os.path.exists(os.path.join(self.data_dir, self.graphfile)):
            self.logger.info("processing GFA file {}".format(os.path.join(self.data_dir, self.graphfile)))
            self.read_gfa()
            self.logger.info(f"read {len(self.edges_src)}, edges")
            np.save(os.path.join(self.cache_dir, "edge_weights.npy"), self.edge_weights)
            scipy.sparse.save_npz(os.path.join(self.cache_dir, "adj_sparse.npz"), self.adj_matrix)
            np.save(os.path.join(self.cache_dir, "graph_nodes.pkl"), self.graph_nodes)
            pickle.dump(self.graph_paths, open(os.path.join(self.cache_dir, "graph_paths.pkl"), 'wb'))
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

    def print_stats(self):
        print("==============================")
        print("DATASET STATS:")
        # get:
        #   number of sequences
        #   number of contigs
        #   length of contigs (sum and average and N50)
        #   coverage samples from jgi file
        #   assembly_graph.file exists, number of edges, number of paths
        print("number of sequences: {}".format(len(self.node_names)))
        print("assembly length: {} Gb".format(round(sum(self.node_lengths) / 1000000000, 3)))
        print("assembly N50: {} Mb".format(round(self.calculate_n50()/1000000, 3)))
        print("assembly average length (Mb): {} max: {} min: {}".format(round(np.mean(self.node_lengths)/1000000, 3),
                                                                   round(np.max(self.node_lengths)/1000000, 3),
                                                                   round(np.min(self.node_lengths)/1000000, 3)))
        print("coverage samples: {}".format(len(self.node_depths[0])))
        if os.path.exists(os.path.join(self.data_dir, self.graphfile)) or \
            len(self.edges_src) > 0:
            print("Graph file found and read")
            print("graph edges: {}".format(len(self.edges_src)))
            print("contig paths: {}".format(len(self.graph_paths)))
        else:
            print("No assembly graph loaded")
        #   contigs with markers on marker_gene_stats
        #   stats with SCGs (max/min # of contigs, etc)
        #   TODO: contigs wiht same SCGs
        if len(self.contig_markers) > 0:
            print("total ref markers sets: {}".format(len(self.ref_marker_sets)))
            print("total ref markers: {}".format(len(self.markers)))
            n_of_markers = [len(x) for x in self.contig_markers.values() if len(x) > 0]
            print("contigs with one or more markers: {}/{}".format(len(n_of_markers),
                                                                    len(self.node_names)))
            
            print("max SCGs on one contig: {}, average(excluding 0): {}".format(max(n_of_markers),
                                                            np.mean(n_of_markers)))
            self.estimate_n_genomes()
            print("SCG contig count min: {} contigs".format(min(self.scg_counts.values())))
        else:
            print("No SCG markers")
        print("==============================")
                                                          
    def calculate_n50(self):
        """Calculate N50 for a sequence of numbers.
    
        Args:
            list_of_lengths (list): List of numbers.
    
        Returns:
            float: N50 value.
        from https://onestopdataanalysis.com/n50-genome/
        """
        tmp = []
        for tmp_number in set(self.node_lengths):
                tmp += [tmp_number] * list(self.node_lengths).count(tmp_number) * tmp_number
        tmp.sort()
    
        if (len(tmp) % 2) == 0:
            median = (tmp[int(len(tmp) / 2) - 1] + tmp[int(len(tmp) / 2)]) / 2
        else:
            median = tmp[int(len(tmp) / 2)]
    
        return median

    def read_cache(self, load_graph=True):
        prefix = os.path.join(self.cache_dir, "{}")
        self.node_names = list(np.load(prefix.format("node_names.npy")))
        self.node_seqs = pickle.load(open(prefix.format("node_seqs.pkl"), "rb"))
        # graph nodes
        self.node_lengths = np.load(prefix.format("node_lengths.npy"))
        if load_graph:
            self.graph_nodes = np.load(prefix.format("graph_nodes.npy"))
            self.graph_paths = pickle.load(open(prefix.format("graph_paths.pkl"), 'rb'))
            self.adj_matrix = scipy.sparse.load_npz(prefix.format("adj_sparse.npz"))
            self.edge_weights = self.adj_matrix.data
            self.edges_src = self.adj_matrix.row
            self.edges_dst = self.adj_matrix.col
        self.node_kmers = np.load(prefix.format("node_attributes_kmer.npy"))
        self.node_depths = np.load(prefix.format("node_attributes_depth.npy"))
        # self.edge_weights = np.load(prefix.format("edge_weights.npy"))
        
        # labels
        self.node_to_label = np.load("{}/node_to_label.npy".format(self.cache_dir), allow_pickle=True)[()]
        self.label_to_node = np.load("{}/label_to_node.npy".format(self.cache_dir), allow_pickle=True)[()]
        self.labels = list(np.load("{}/labels.npy".format(self.cache_dir)))

    def check_cache(self, require_graph=True):
        """check if all necessary files exist in cache"""
        prefix = os.path.join(self.cache_dir, "{}")
        if require_graph:
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
        else:
            return (
                os.path.exists(prefix.format("node_names.npy"))
                and os.path.exists(prefix.format("node_lengths.npy"))
                and os.path.exists(prefix.format("node_attributes_kmer.npy"))
                and os.path.exists(prefix.format("node_attributes_depth.npy"))
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
                elif line.startswith("L"):  # link/edge
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
                elif line.startswith("P"):
                    values = line.strip().split()  # P contig path
                    self.graph_paths[values[1]] = [self.node_names.index(path_node_name[:-1]) \
                        for path_node_name in values[2].split(",")]
        self.logger.info(f"skipped contigs {len(skipped_contigs)} < {self.min_contig_length}")
        self.adj_matrix = scipy.sparse.coo_matrix(
            (self.edge_weights, (self.edges_src, self.edges_dst)), shape=(len(self.node_names), len(self.node_names))
        )
        self.edge_weights = np.array(self.edge_weights)

    def read_depths(self):
        node_depths = {}
        if self.depthfile is not None:
            if self.depthfile.endswith(".npz"):
                self.node_depths = np.load(open(os.path.join(self.data_dir, self.depthfile), "rb"))["arr_0"]
                if self.node_depths.shape[0] != len(self.node_names):
                    print("depth npz file mismatch:")
                    breakpoint()
            else:
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
                depthssum = self.node_depths.sum(axis=1) + 1e-10
                self.node_depths /= depthssum.reshape((-1, 1))
            else:
                self.node_depths = stats.zscore(self.node_depths, axis=0)
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
            if len(self.edge_weights) > 1:
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
            scg_path = os.path.join(self.data_dir, self.scgfile)
            ref_sets = read_marker_gene_sets(BACTERIA_MARKERS)
            contig_markers = read_contig_genes(scg_path)
            self.ref_marker_sets = ref_sets
            self.contig_markers = contig_markers
            marker_counts = get_markers_to_contigs(ref_sets, contig_markers)
            self.markers = marker_counts
        else:
            self.ref_marker_sets = {}
            self.contig_markers = {}
            self.run_checkm()

    def get_all_different_idx(self):
        """
        Returns a 2d numpy array where each row
        corresponds to a pairs of node idx whose
        feature must be different as they correspond
        to the same contig (check jargon). This
        should encourage the HQ value to be higher.
        """
        node_names_to_idx = {node_name: i for i, node_name in enumerate(self.node_names)}
        pair_idx = set()
        for n1 in self.contig_markers:
            for gene1 in self.contig_markers[n1]:
                for n2 in self.contig_markers:
                    if n1 != n2 and gene1 in self.contig_markers[n2]:
                        p1 = (node_names_to_idx[n1], node_names_to_idx[n2])
                        p2 = (node_names_to_idx[n2], node_names_to_idx[n1])
                        if (p1 not in pair_idx) and (p2 not in pair_idx):
                            pair_idx.add(p1)
        pair_idx = np.unique(np.array(list(pair_idx)), axis=0)
        print("Number of diff cluster pairs:", len(pair_idx))
        self.neg_pairs_idx = pair_idx

    def run_vamb(self, vamb_outdir, cuda, vambdim):
        from vamb.vamb_run import run as run_vamb
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
        if self.featuresfile.endswith(".tsv"):
            with open(self.featuresfile, "r") as ffile:
                for line in ffile:
                    values = line.strip().split()
                    node_embs[values[0]] = [float(x) for x in values[1:]]
            self.logger.info("loaded {} features/ {} nodes from tsv".format(len(node_embs), len(self.node_names)))
        elif self.featuresfile.endswith(".pickle"):
            with open(self.featuresfile, "rb") as ffile:
                node_embs = pickle.load(ffile)
            self.logger.info("loaded {} features/ {} nodes from pickle".format(len(node_embs), len(self.node_names)))
        self.node_embs = [
                node_embs.get(n, np.random.uniform(10e-5, 1.0, len(node_embs[list(node_embs.keys())[0]]))) for n in self.node_names
            ]  # deal with missing embs
        self.node_embs = np.array(self.node_embs)
        

    def get_topk_neighbors(self, k, scg_only=False):
        """
        Returns a list of the top k neighbors for each node. Use kmers and abundance

        """
        #breakpoint()
        self.logger.info("getting top {} neighbors".format(k))
        self.topk_neighbors = []
        features = np.concatenate((self.node_kmers, self.node_depths), axis=1)
        cosine_dists = np.dot(features, features.T)
        for i in range(len(self.node_names)):
            self.topk_neighbors.append(set(np.argsort(cosine_dists[i])[-k:]))
        self.logger.info("got top {} neighbors".format(k))

    def estimate_n_genomes(self):
        self.scg_counts = {}
        for marker_set in self.ref_marker_sets:
            for gene in marker_set:
                self.scg_counts[gene] = 0
                for contig in self.contig_markers:
                    if gene in self.contig_markers[contig]:
                        self.scg_counts[gene] += self.contig_markers[contig][gene]

        #print(self.scg_counts)
        quartiles = np.percentile(list(self.scg_counts.values()), [25, 50, 75])
        print("candidate k0s", sorted(set([k for k in self.scg_counts.values() if k >= quartiles[2]])))
        return max(self.scg_counts.values())
        
       
    def run_checkm(self, nthreads=10, tempdir="../temp"):
        # check if checkm is installed
        checkm_is_avail = shutil.which("checkm") is not None
        # if not, print commands only
        commands = [
            "mkdir {}/nodes".format(self.data_dir), #; cd nodes; ",
            "cat {0}/{1} | awk '{{ if (substr($0, 1, 1)=='>') {{filename=(substr($0,2) '.fa')}} print $0 > {0}/filename }}".format(self.data_dir, self.fastafile),
            'find {}/nodes/ -name "* *" -type f | rename "s/ /_/g"'.format(self.data_dir),
            "checkm taxonomy_wf --tmpdir {0} -t {1} -x fa domain Bacteria {2}/nodes/ {2}/checkm_nodes/".format(tempdir, nthreads, self.data_dir)
        ]
        if checkm_is_avail:
            #run commands one by one
            for cmd in commands:
                os.system(cmd)
        else:
            print("Run this on a machine with CheckM installed")
            for cmd in commands:
                print(cmd)
        

