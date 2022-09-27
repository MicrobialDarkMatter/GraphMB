import os
import torch
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.data import DGLDataset
import networkx as nx

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
            [self.assembly.labels.index(self.assembly.node_to_label[n]) if self.assembly.node_to_label[n] in self.assembly.labels else 0 for n in self.assembly.node_names]
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
