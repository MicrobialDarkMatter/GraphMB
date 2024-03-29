import numpy as np
import datetime
import os
import sys
import tensorflow as tf
import random
import logging
from tqdm import tqdm
import scipy
from graph_functions import set_seed, run_tsne, plot_embs, plot_edges_sim, draw_nx_graph
from graphmb.evaluate import calculate_overall_prf
from vaegbin import name_to_model, TensorboardLogger, prepare_data_for_gnn, compute_clusters_and_stats, \
                              log_to_tensorboard, eval_epoch_cluster, eval_epoch, compute_cluster_score
from models import VAEDecoder, VAEEncoder

class THAug:
    def __init__(self, encoder, decoder, gnn_model, num_negatives=10, lr_gnn=1e-3,  lr_ae=1e-2, kld_weight=1/200.,
                scg_weight=100, labels=None):
        self.encoder = encoder
        self.decoder = decoder
        self.gnn_model = gnn_model
        self.kld_weight = kld_weight
        self.abundance_dim = self.encoder.abundance_dim
        self.kmers_dim = self.encoder.kmers_dim
        self.num_negatives = num_negatives
        self.scg_weight = scg_weight

        if self.abundance_dim > 1:
            self.abundance_weight = 0.85 / tf.math.log(float(self.abundance_dim))
            self.kmer_weight = 0.15
        else:
            self.abundance_weight = 0.5
            self.kmer_weight = 0.5
        self.opt_vae = tf.keras.optimizers.Adam(learning_rate=lr_ae)
        self.opt_gnn = tf.keras.optimizers.Adam(learning_rate=lr_gnn)
        self.logvar = None
        self.mu = None
        self.labels = labels

        self.adj_shape = self.gnn_model.adj.dense_shape
        S = tf.cast(tf.reduce_sum(self.gnn_model.adj.values), tf.float32)
        s0 = tf.cast(self.adj_shape[0], tf.float32)
        self.pos_weight = (s0 * s0 - S) / S
        self.norm = s0 * s0 / ((s0 * s0 - S) * 2.0)
    
    @tf.function
    def train_supervised(self, X, idx):
        with tf.GradientTape() as tape:
            y_hat = self.gnn_model(X, idx)
            y = tf.gather(self.gnn_model.labels, idx)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_hat, from_logits=True)
            loss = tf.reduce_mean(loss)
        tw = self.gnn_model.trainable_weights
        grads = tape.gradient(loss, tw)
        self.opt_gnn.apply_gradients(zip(grads, tw))
        return loss

    def train_vae_step(self, x, agraph, writer=None, epoch=0, vae=True):
        #breakpoint()
        losses, z = self._train_step(x, agraph, vae=vae, writer=writer, epoch=epoch)
        if writer is not None:
            with writer.as_default():
                tf.summary.scalar('mean logvar', self.logvar, step=epoch)
                tf.summary.scalar('mean mu', self.mu, step=epoch)
        losses = [loss.numpy() for loss in losses]
        return losses, z

    def get_vae_edges(self, z, th=0.9, batch_size=128):
        # split into batches of size batch_size
        idx = list(range(z.shape[0]))
        n_batches = len(idx)//batch_size
        batches = [idx[i:i+batch_size] for i in range(0, len(idx), batch_size)]
        norm_z = tf.nn.l2_normalize(z, axis=1)
        # for each batch, get the edges
        # TODO use relu as in Dai 2022
        #edges = tf.sparse.SparseTensor(np.empty((0, 2), dtype=np.int64), [], (norm_z.shape[0], norm_z.shape[0]))
        edges = tf.zeros([0,2], dtype=tf.int64)
        for i, batch in enumerate(batches):
            # (bs,z) x (z * n) -> bs,n
            dists = tf.keras.activations.relu(tf.matmul(tf.gather(norm_z,batch), norm_z, transpose_b=True))
            #dists = tf.nn.sigmoid(dists)
            #dists = dists / tf.reduce_max(dists)
            filt_dists = tf.math.greater_equal(dists, th)
            #out = tf.cast(out, tf.int32)
            batch_edges = tf.where(filt_dists) 
            batch_edges_indices = batch_edges + [batch_size*i, 0]
            edges = tf.concat([edges, batch_edges_indices], axis=0)
            #batch_edges_sparse = tf.sparse.SparseTensor(batch_edges_indices, dists[filt_dists], (z.shape[0], z.shape[0]))
            #edges = tf.sparse.add(edges, batch_edges_sparse)
            #edges += batch_edges.numpy().tolist()
        return edges

    def get_vae_edges_neighbors(self, z, agraph_n, feat_n, th=0.9):
        # split into batches of size batch_size
        idx = list(range(z.shape[0]))
        norm_z = tf.nn.l2_normalize(z, axis=1)
        # for each batch, get the edges
        # use relu as in Dai 2022
        #edges = tf.sparse.SparseTensor(np.empty((0, 2), dtype=np.int64), [], (norm_z.shape[0], norm_z.shape[0]))
        candidates = tf.concat([agraph_n, feat_n], axis=0)
        row_embs = tf.gather(indices=candidates[:, 0], params=norm_z)
        col_embs = tf.gather(indices=candidates[:, 1], params=norm_z)
        dists = tf.keras.activations.relu(tf.reduce_sum(tf.math.multiply(row_embs, col_embs), axis=1))
        filt_dists = tf.math.greater_equal(dists, th)
        #out = tf.cast(out, tf.int32)
        edges = candidates[filt_dists]
        return edges

    
    @tf.function
    def loss(self, x, mu, logvar, vae, training=True, writer=None, epoch=0):
        if vae:
            epsilon = tf.random.normal(tf.shape(mu))
            z = mu + epsilon * tf.math.exp(0.5 * logvar)
            kld  = 0.5*tf.math.reduce_mean(tf.math.reduce_mean(1.0 + logvar - tf.math.pow(mu, 2) - tf.math.exp(logvar), axis=1))
            kld  = kld * self.kld_weight
        else:
            z = mu
            kld = 0
        x_hat = self.decoder(z, training=training)
        if writer is not None:
            with writer.as_default():
                tf.summary.scalar('min ab', tf.reduce_min(x_hat[:, :self.abundance_dim]), step=epoch)
        if self.abundance_dim > 1:
            mse1 = - tf.reduce_mean(tf.reduce_sum((tf.math.log(x_hat[:, :self.abundance_dim] + 1e-9) * x[:, :self.abundance_dim]), axis=1))
        else:
            mse1 = self.abundance_weight*tf.reduce_mean( (x[:, :self.abundance_dim] - x_hat[:, :self.abundance_dim])**2)
        mse2 = self.kmer_weight*tf.reduce_mean( tf.reduce_mean((x[:, self.abundance_dim:] - x_hat[:, self.abundance_dim:])**2, axis=1))
        #self.z = z
        return mse1, mse2, kld, z

    @tf.function
    def edge_predict_loss(agraph, node_embs, node_features, sigma=100):
        """
        BCE on exp(SSE)*ReLU(z*zT)
        """
        # pos loss
        breakpoint()
        norm_z = tf.nn.l2_normalize(node_embs, axis=1)
        row_embs = tf.gather(indices=agraph.indices[:, 0], params=norm_z)
        col_embs = tf.gather(indices=agraph.indices[:, 1], params=norm_z)
        pos_dists = tf.keras.activations.relu(tf.reduce_sum(tf.math.multiply(row_embs, col_embs), axis=1))
        pos_loss = tf.math.power(pos_dists - 1, 2)
        pos_weights = tf.gather(indices=agraph.indices[:, 0], params=node_features) - \
                                  tf.gather(indices=agraph.indices[:, 1], params=node_features)
        pos_weights = tf.math.exp(-tf.math.power(pos_weights, 2)/tf.math.power(sigma, 2))
        pos_loss = tf.reduce_mean(pos_loss * pos_weights)
        #sample negatives
        return pos_loss

    @tf.function
    def _train_step(self, x, agraph, vae=True, writer=None, epoch=0):
        with tf.GradientTape() as tape:
            mu, logvar = self.encoder(x, training=True)
            logvar = tf.clip_by_value(logvar, -2, 2)
            self.logvar = tf.cast(tf.math.reduce_mean(logvar), float)
            self.mu = tf.cast(tf.math.reduce_mean(mu), float)
            mse1, mse2, kld, z = self.loss(x, mu, logvar, vae, training=True, writer=writer, epoch=epoch)
            loss = mse1 + mse2 - kld

            graph_loss = self.edge_predict_loss(agraph, mu, x)

        tw = self.encoder.trainable_weights + self.decoder.trainable_weights   
        grads = tape.gradient(loss, tw)
        grad_norm = tf.linalg.global_norm(grads)
        clip_grads, _ = tf.clip_by_global_norm(grads, 5,  use_norm=grad_norm)
        new_grad_norm = tf.linalg.global_norm(clip_grads)
        self.opt_vae.apply_gradients(zip(clip_grads, tw))

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar('grad norm', grad_norm, step=epoch)
                tf.summary.scalar('clipped grad norm', new_grad_norm, step=epoch)

        return (loss, mse2, mse1, kld), z

    @tf.function
    def train_unsupervised(self, x, idx):
        with tf.GradientTape() as tape:
            #breakpoint()
            # run encoder first
            # run gnn model
            node_hat = self.gnn_model(x, idx)
            gnn_loss = tf.constant(0, dtype=tf.float32)

            # create random negatives for gnn_loss
            row_embs = tf.gather(indices=self.gnn_model.adj.indices[:, 0], params=node_hat)
            col_embs = tf.gather(indices=self.gnn_model.adj.indices[:, 1], params=node_hat)
            positive_pairwise = tf.reduce_sum(tf.math.multiply(row_embs, col_embs), axis=1)

            neg_idx = tf.random.uniform(
                shape=(self.num_negatives * len(self.gnn_model.adj.indices),),
                minval=0,
                maxval=self.adj_shape[0] * self.adj_shape[1] - 1,
                dtype=tf.int64,
            )
            neg_idx_row = tf.math.minimum(
                tf.cast(self.adj_shape[0] - 1, tf.float32),
                tf.cast(neg_idx, tf.float32) / tf.cast(self.adj_shape[1], tf.float32),
            )
            neg_idx_row = tf.cast(neg_idx_row, tf.int64)
            neg_idx_col = tf.cast(tf.math.minimum(self.adj_shape[1] - 1, (neg_idx % self.adj_shape[1])), tf.int64)
            neg_idx = tf.concat((neg_idx_row, neg_idx_col), axis=-1)
            try:
                #negative_pairs = tf.gather_nd(pairwise_similarity, neg_idx)
                neg_row_embs = tf.gather(indices=neg_idx_row, params=node_hat)
                neg_col_embs = tf.gather(indices=neg_idx_col, params=node_hat)
                negative_pairs = tf.reduce_sum(tf.math.multiply(neg_row_embs, neg_col_embs), axis=1)
            except:
                breakpoint()

            pos_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(positive_pairwise),
                                                               positive_pairwise, from_logits=True)
            neg_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(negative_pairs),
                                                            negative_pairs, from_logits=True)
            gnn_loss = 0.5 * (pos_loss + neg_loss) #* self.gnn_weight
            #breakpoint()
            #y_true = tf.concat((tf.ones_like(positive_pairwise), tf.ones_like(negative_pairs)), axis=0)
            #y_pred = tf.concat((positive_pairwise, negative_pairs), axis=0)
            #gnn_loss = tf.keras.metrics.binary_crossentropy(y_true, y_pred, from_logits=True)
            #gnn_loss = gnn_loss # * self.gnn_weight
            loss = gnn_loss
            
            # SCG loss
            scg_loss = tf.constant(0, dtype=tf.float32)
            """if self.all_different_idx is not None and self.scg_weight > 0:
                ns1 = tf.gather(node_hat, self.all_different_idx[:, 0])
                ns2 = tf.gather(node_hat, self.all_different_idx[:, 1])
                all_diff_pairs = tf.math.exp(-0.5 * tf.reduce_sum((ns1 - ns2) ** 2, axis=-1))
                scg_loss = tf.reduce_mean(all_diff_pairs) * self.scg_weight
                loss += scg_loss"""

        tw = self.gnn_model.trainable_weights
        grads = tape.gradient(loss, tw)
        self.opt_gnn.apply_gradients(zip(grads, tw))
        return loss, gnn_loss, scg_loss


def draw_graph_with_labels(dataset, adj, cluster_labels, outname):
    #th.gnn_model.adj
    #f"{args.outdir}/{args.outname}_vaeembs_epoch{e}"
    node_to_label = {i: l for i, l in enumerate(cluster_labels)}
    label_to_node = {s: [] for s in set(cluster_labels)}
    for n in node_to_label:
        s = node_to_label[n]
        label_to_node[s].append(n)
    cluster_stats = compute_cluster_score(dataset.ref_marker_sets, dataset.contig_markers, np.array(dataset.node_names), cluster_labels)
    cluster_info = {label: "{:.2f}/{:.2f}/{}".format(cluster_stats[label]['comp'], cluster_stats[label]['cont'], len(label_to_node[label])) for label in label_to_node}
    # plot graph
    csr_graph = scipy.sparse.csr_array((adj.values, (adj.indices[:,0], adj.indices[:,1])))
    csr_graph.setdiag(0)
    draw_nx_graph(csr_graph, node_to_label, label_to_node, basename=outname, contig_sizes=dataset.node_lengths, node_titles=None, cluster_info=cluster_info)

def update_graph(adj, positive_pairs):
     # remove edges from a graph with nodes from positive_pairsq
    #breakpoint()
    #remove self edges
    not_self_nodes = positive_pairs[positive_pairs[:, 0] != positive_pairs[:, 1]]
    positive_nodes = np.unique(not_self_nodes.flatten())
    #th.gnn_model.adj = adj
    retain = [i not in positive_nodes and j not in positive_nodes for (i, j) in adj.indices]
    removed_edges = len(adj.indices) - sum(retain)
    logging.info("removing {} edges to graph of {} edges".format(removed_edges, len(adj.indices)))
    new_adj = tf.sparse.retain(adj, retain)
    new_edges_sparse = tf.sparse.SparseTensor(positive_pairs, np.full(len(positive_pairs), fill_value=2.0, dtype=np.float32), (adj.shape[0], adj.shape[0]))
    logging.info("adding {} edges to graph of {} edges".format(len(new_edges_sparse.indices), len(new_adj.indices)))
    new_adj = tf.sparse.add(new_adj, new_edges_sparse)
    logging.info("graph now has {} edges".format(len(new_adj.indices)))
    return new_adj


def run_model_vaegnn(dataset, args, logger, nrun):
    set_seed(args.seed)
    node_names = np.array(dataset.node_names)
    RESULT_EVERY = args.evalepochs
    hidden_gnn = args.hidden_gnn
    hidden_vae = args.hidden_vae
    output_dim_gnn = args.embsize_gnn
    output_dim_vae = args.embsize_vae
    epochs = args.epoch
    lr_vae = args.lr_vae
    lr_gnn = args.lr_gnn
    nlayers_gnn = args.layers_gnn
    gname = args.model_name
    if gname == "vae":
        args.ae_only = True
    else:
        gmodel_type = name_to_model[gname.split("_")[0].upper()]
    clustering = args.clusteringalgo
    #k = args.kclusters
    if args.kclusters is None:
        k = dataset.estimate_n_genomes()
    else:
        k = args.kclusters
    use_edge_weights = True
    use_disconnected = not args.quick
    cluster_markers_only = args.quick
    decay = 0.5 ** (2.0 / epochs)
    concat_features = args.concat_features
    use_ae = gname.endswith("_ae") or args.ae_only or gname == "vae"

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(args.outdir, 'logs/' + args.outname + current_time + '/train')
    summary_writer = tf.summary.create_file_writer(train_log_dir)
    print("logging to tensorboard")
    tb_handler = TensorboardLogger(summary_writer, runname=args.outname + current_time)
    logger.addHandler(tb_handler)
    #tf.summary.trace_on(graph=True)

    logger.info("******* Running model: {} **********".format(gname))
    logger.info("***** using edge weights: {} ******".format(use_edge_weights))
    logger.info("***** using disconnected: {} ******".format(use_disconnected))
    logger.info("***** concat features: {} *****".format(concat_features))
    logger.info("***** cluster markers only: {} *****".format(cluster_markers_only))
    logger.info("***** threshold adj matrix: {} *****".format(args.binarize))
    logger.info("***** self edges only: {} *****".format(args.noedges))
    args.rawfeatures = True
    logger.info("***** Using raw kmer+abund features: {}".format(args.rawfeatures))
    tf.config.experimental_run_functions_eagerly(True)


    X, adj, cluster_mask, neg_pair_idx, pos_pair_idx = prepare_data_for_gnn(
            dataset, use_edge_weights, cluster_markers_only, use_raw=args.rawfeatures,
            binarize=args.binarize, remove_edges=args.noedges)
    logger.info("***** SCG neg pairs: {}".format(neg_pair_idx.shape))
    logger.info("***** input features dimension: {}".format(X[cluster_mask].shape))
    # pre train clustering
    if not args.skip_preclustering:
        cluster_labels, stats, _, hq_bins = compute_clusters_and_stats(
                    X[cluster_mask], node_names[cluster_mask],
                    dataset, clustering=clustering, k=k, tsne=args.tsne, #cuda=args.cuda,
                )
        logger.info(f">>> Pre train stats: {str(stats)}")
    else:
        stats = {"hq": 0, "epoch":0 }
    
    pname = ""

    #plot edges vs initial embs
    id_to_scg = {i: set(dataset.contig_markers[node_name].keys()) for i, node_name in enumerate(dataset.node_names)}
    plot_edges_sim(X, dataset.adj_matrix, id_to_scg, f"{args.outdir}/{args.outname}_pretrain_")

    scores = [stats]
    losses = {"total": [], "ae": [], "gnn": [], "scg": []}
    all_cluster_labels = []
    X = X.astype(np.float32)
    features = tf.constant(X)
    input_dim_gnn = features.shape[1]

    logger.info(f"*** Model input dim {X.shape[1]}, GNN input dim {input_dim_gnn}, use_ae: {use_ae}, run AE only: {args.ae_only}")
    
    S = []

    gnn_model = gmodel_type(
        features_shape=features.shape,
        input_dim=input_dim_gnn,
        labels=None,
        adj=adj,
        embsize=output_dim_gnn,
        hidden_units=hidden_gnn,
        layers=nlayers_gnn,
        conv_last=False,
        use_bn=True, use_vae=False,
        predict=False,
        )
    logger.info(f"*** output clustering dim {output_dim_gnn}")

    encoder = VAEEncoder(dataset.node_depths.shape[1], dataset.node_kmers.shape[1],
                         hidden_vae, zdim=output_dim_vae, dropout=args.dropout_vae)
    decoder = VAEDecoder(dataset.node_depths.shape[1], dataset.node_kmers.shape[1],
                         hidden_vae, zdim=output_dim_vae, dropout=args.dropout_vae)

    th = THAug(
        gnn_model=gnn_model,
        lr_gnn=lr_gnn,
        lr_ae=lr_vae,
        encoder=encoder,
        decoder=decoder,
    )

    dataset.get_topk_neighbors(k=100)
    neighbors = [(n, m) for n in range(len(dataset.node_names)) for m in dataset.topk_neighbors[n]]

    #gnn_model.summary()
    graph_freq = 10

    if args.eval_split == 0:
        train_idx = np.arange(len(features))
        eval_idx = []
    else:
        train_idx = np.array(random.sample(list(range(len(features))), int(len(features)*(1-args.eval_split))))
        eval_idx = np.array([x for x in np.arange(len(features)) if x not in train_idx])
        logging.info(f"**** using {len(train_idx)} for training and {len(eval_idx)} for eval")
    features = np.array(features)
    pbar_epoch = tqdm(range(epochs), disable=args.quiet, position=0)
    scores = [stats]
    best_embs, best_vae_embs, best_model, best_hq, best_epoch = None, None, th.gnn_model.get_weights(), 0, 0
    batch_size = args.batchsize
    if batch_size == 0:
        batch_size = len(train_idx)
        logger.info("**** initial batch size: {} ****".format(batch_size))
    batch_steps = [25, 75, 150, 300]
    batch_steps = [x for i, x in enumerate(batch_steps) if (2 ** (i+1))*batch_size < len(train_idx)]
    logger.info("**** epoch batch size doubles: {} ****".format(str(batch_steps)))
    vae_losses = []
    step = 0
    gnn_loss, diff_loss = 0, 0
    # draw Assembly graph
    node_to_label = {i: "0" for i, l in enumerate(dataset.node_names)}
    label_to_node = {"0": list(range(len(dataset.node_names)))}
    csr_graph = scipy.sparse.csr_array((th.gnn_model.adj.values, (th.gnn_model.adj.indices[:,0], th.gnn_model.adj.indices[:,1])))
    draw_nx_graph(csr_graph, node_to_label, label_to_node, f"{args.outdir}/{args.outname}_assemblygraph", contig_sizes=dataset.node_lengths, node_titles=None)
    for e in pbar_epoch:
        vae_epoch_losses_list = {"kld": [], "total vae loss": [], "kmer": [], "ab": []}
        np.random.shuffle(train_idx)
        recon_loss = 0

        if e < epochs // 2: #args.evalskip + 50:
            # train VAE in batches
            if e in batch_steps:
                #print(f'Increasing batch size from {batch_size:d} to {batch_size*2:d}')
                batch_size = batch_size * 2
            np.random.shuffle(train_idx)
            n_batches = (len(train_idx)//batch_size) + 1
            pbar_vaebatch = tqdm(range(n_batches), disable=(args.quiet or batch_size == len(train_idx) or n_batches < 100),
                                position=1, ascii=' =')
            # 1: train VAE until epoch 400
            full_z = tf.zeros([0, output_dim_vae])
            for b in pbar_vaebatch:
                
                batch_idx = train_idx[b*batch_size:(b+1)*batch_size]
                vae_losses, batch_z = th.train_vae_step(X[batch_idx], th.gnn_model.adj, summary_writer, step, vae=True)
                vae_epoch_losses_list["total vae loss"].append(vae_losses[0])
                vae_epoch_losses_list["kmer"].append(vae_losses[1])
                vae_epoch_losses_list["ab"].append(vae_losses[2])
                vae_epoch_losses_list["kld"].append(vae_losses[3])
                pbar_vaebatch.set_description(f'E={e} L={np.mean(vae_epoch_losses_list["total vae loss"][-10:]):.4f}')
                step += 1
                full_z = tf.concat([full_z, batch_z], axis=0)
            assert full_z.shape[0] == len(train_idx)

            vae_epoch_losses = {k: np.mean(v) for k, v in vae_epoch_losses_list.items()}
            log_to_tensorboard(summary_writer, vae_epoch_losses, step)


            eval_loss, eval_mse1, eval_mse2, eval_kld = 0, 0, 0, 0
            recon_loss = np.mean(vae_epoch_losses["total vae loss"])
        else:
            vae_epoch_losses = {"kld": 0, "total vae loss": 0, "kmer": 0, "ab": 0}
        #th.encoder = th_vae.encoder
        if e > epochs // 2: #args.evalskip + 50: # and e % graph_freq == 0: # Train graph embs
        # Calculate dist matrix
            if e % graph_freq == 0:
                th.z = th.encoder(X)[0]
                #dist_matrix = th.get_vae_edges(full_z, 0.9)
                new_edges = th.get_vae_edges_neighbors(full_z, th.gnn_model.adj.indices,
                                                       tf.convert_to_tensor(neighbors, dtype=tf.int64), 0.1)
                #breakpoint()
                #logging.info("adding {} edges to graph of {} edges".format(len(dist_matrix), len(th.gnn_model.adj.indices)))
                # merge with Assembly graph
                #th.gnn_model.adj = tf.sparse.add(adj, dist_matrix)
                #logging.info("graph now has {} edges".format(len(th.gnn_model.adj.indices)))
                #th.gnn_model.adj = update_graph(th.gnn_model.adj, dist_matrix.numpy())
                th.gnn_model.adj =  tf.sparse.SparseTensor(new_edges, np.full(len(new_edges), fill_value=2.0, dtype=np.float32), (adj.shape[0], adj.shape[0]))

            # Train GNN
            #total_loss, gnn_loss, diff_loss = th.train_unsupervised(X, train_idx)
            #breakpoint()

            th.gnn_model.labels = np.array([hq_bins.index(n)+1 if n in hq_bins else 0 for n in cluster_labels])
            total_loss = th.train_supervised(X, train_idx)
            gnn_loss, diff_loss = total_loss, 0
            #log_to_tensorboard(summary_writer, {"Total loss": total_loss, "gnn loss": gnn_loss, "SCG loss": diff_loss,
            #                                    'GNN LR': th.opt_gnn.learning_rate}, e)
            log_to_tensorboard(summary_writer, {"Total loss": total_loss,
                                                'GNN LR': th.opt_gnn.learning_rate}, step)
            #gnn_loss = gnn_loss.numpy()
            #diff_loss = diff_loss.numpy()
        #else:
            
            

        #gpu_mem_alloc = tf.config.experimental.get_memory_info('GPU:0')["peak"] / 1000000 if args.cuda else 0
        ## EVAL ###
        gpu_mem_alloc = tf.config.experimental.get_memory_usage('GPU:0') / 1000000 if args.cuda else 0
        if (e + 1) % RESULT_EVERY == 0 and e > args.evalskip:
            gnn_input_features = features
            #if use_ae:
            gnn_input_features = th.encoder(features)[0]
            #if not args.ae_only:
            # use the VAE embs for the first half and then the GNN embs
            if e > epochs // 2:
                node_new_features = th.gnn_model(features, None, training=False).numpy()
                 #gnn_input_features = gnn_input_features.numpy()
            #    node_new_features = node_new_features.numpy()
            else:
                node_new_features = gnn_input_features.numpy()
            
            if concat_features:
                node_new_features = tf.concat([gnn_input_features, node_new_features], axis=1).numpy()

            
            weights = th.gnn_model.get_weights()
            #best_hq, best_embs, best_epoch, scores, best_model = eval_epoch(logger, summary_writer, node_new_features,
            #                                                    cluster_mask, weights, step, args, dataset, e, scores,
            #                                                    best_hq, best_embs, best_epoch, best_model)
            stats, new_best, cluster_labels, positive_pairs = eval_epoch_cluster(logger=logger,
                                                                                    summary_writer=summary_writer,
                                                                                    node_new_features=node_new_features,
                                                                                    cluster_mask=cluster_mask,
                                                                                    best_hq=best_hq,
                                                                                    step=step, args=args,
                                                                                    dataset=dataset, epoch=e,
                                                                                    tsne=False, cluster=True)
            scores.append(stats)
            if new_best:
                best_hq = stats["hq"]
                best_embs = node_new_features
                best_epoch = e
            
            
            
            draw_graph_with_labels(dataset, adj, cluster_labels, f"{args.outdir}/{args.outname}_assemblygraph_epoch{e}")

            # Update graph
            if e < epochs // 2:
                #gnn_input_features = gnn_input_features.numpy()
                #stats, new_best, cluster_labels, positive_pairs = eval_epoch_cluster(logger=logger,
                #                                                                    summary_writer=summary_writer,
                #                                                                    node_new_features=gnn_input_features,
                #                                                                    cluster_mask=cluster_mask,
                #                                                                    best_hq=best_hq,
                #                                                                    step=step, args=args,
                #                                                                    dataset=dataset, epoch=e,
                #                                                                    tsne=False)
                
                
                
                print("vae stats", stats)
                # add edges to graph
                #breakpoint()
                #th.gnn_model.adj = update_graph(th.gnn_model.adj, positive_pairs)
                draw_graph_with_labels(dataset, th.gnn_model.adj, cluster_labels, f"{args.outdir}/{args.outname}_currentgraph_vaeembs_epoch{e}")
                #breakpoint()
            else:
                draw_graph_with_labels(dataset, th.gnn_model.adj, cluster_labels, f"{args.outdir}/{args.outname}_currentgraph_gnnembs_epoch{e}")
                

        losses_string = " ".join([f"{k}={v:.3f}" for k, v in vae_epoch_losses.items()])
        pbar_epoch.set_description(
            f"[{args.outname} {nlayers_gnn}l {pname}] GNN={gnn_loss:.3f} SCG={diff_loss:.3f} TotalVAE={recon_loss:.3f} {losses_string} HQ={scores[-1]['hq']}  BestHQ={best_hq} Best Epoch={best_epoch} Max GPU MB={gpu_mem_alloc:.1f}"
        )
        log_to_tensorboard(summary_writer, {"hq_bins": stats["hq"], "mq_bins": stats["mq"]}, step)
        total_loss = gnn_loss + diff_loss + recon_loss
        losses["gnn"].append(gnn_loss)
        losses["scg"].append(diff_loss)
        losses["ae"].append(recon_loss)
        losses["total"].append(total_loss)
        #if e == 1:
            #breakpoint()
        #    with summary_writer.as_default():
        #        tf.summary.trace_export(args.outname, step=0, profiler_outdir=train_log_dir) 
        #        summary_writer.flush()
    if best_embs is None:
        best_embs = node_new_features
    
    cluster_labels, stats, _, _ = compute_clusters_and_stats(
        best_embs, node_names, dataset, clustering=clustering, k=k,
        #cuda=args.cuda,
    )

    
    stats["epoch"] = e
    scores.append(stats)
    # get best stats:
    # if concat_features:  # use HQ
    hqs = [s["hq"] for s in scores]
    #epoch_hqs = [s["epoch"] for s in scores]
    best_idx = np.argmax(hqs)
    # else:  # use F1
    #    f1s = [s["f1"] for s in scores]
    #    best_idx = np.argmax(f1s)
    # S.append(stats)
    S.append(scores[best_idx])
    logger.info(f">>> best epoch all contigs: {RESULT_EVERY + (best_idx*RESULT_EVERY)} : {stats} <<<")
    logger.info(f">>> best epoch: {RESULT_EVERY + (best_idx*RESULT_EVERY)} : {scores[best_idx]} <<<")
    with open(f"{dataset.name}_{gname}_{clustering}{k}_{nlayers_gnn}l_{pname}_results.tsv", "w") as f:
        f.write("@Version:0.9.0\n@SampleID:SAMPLEID\n@@SEQUENCEID\tBINID\n")
        for i in range(len(cluster_labels)):
            f.write(f"{node_names[i]}\t{cluster_labels[i]}\n")
    #del gnn_model, th
    # res_table.add_row(f"{gname} {clustering}{k} {nlayers}l {pname}", S)
    # if gt_idx_label_to_node is not None:
    #    # save embs
    #    np.save(f"{dataset}_{gname}_{clustering}{k}_{nlayers}l_{pname}_embs.npy", node_new_features)
    # plot_clusters(node_new_features, features.numpy(), cluster_labels, f"{gname} VAMB {nlayers}l {pname}")
    # res_table.show()
    #breakpoint()
    #plt.plot(range(len(losses["total"])), losses["total"], label="total loss")

    #plot edges vs initial embs
    #plot_edges_sim(best_vae_embs, dataset.adj_matrix, id_to_scg, "vae_")
    plot_edges_sim(best_embs, dataset.adj_matrix, id_to_scg, f"{args.outdir}/{args.outname}_posttrain_")
    return best_embs, scores[best_idx]