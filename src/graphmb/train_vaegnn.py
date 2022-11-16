import numpy as np
import datetime
import os
import tensorflow as tf
import random
import logging
from tqdm import tqdm
import mlflow
from scipy.sparse import coo_matrix, diags

from graphmb.models import  TH, TrainHelperVAE, VAEDecoder, VAEEncoder, LabelClassifier
from graphmb.utils import set_seed
from graphmb.visualize import plot_edges_sim
from graphmb.evaluate import compute_clusters_and_stats, eval_epoch
from graphmb.utils import get_cluster_mask
from graphmb.gnn_models import name_to_model
    
class TensorboardLogger(logging.StreamHandler):
    def __init__(self, file_writer, runname=""):
        logging.StreamHandler.__init__(self)
        self.file_writer = file_writer
        self.runname = runname
        self.step = 0

    def emit(self, msg):
        with self.file_writer.as_default():
            tf.summary.text(self.runname, msg.msg, step=self.step)
        self.step += 1


def log_to_tensorboard(writer, values, step):
    """Write key-values to writer
    """
    for k, v in values.items():
        with writer.as_default():
            tf.summary.scalar(k, v, step=step)


def normalize_adj_sparse(A):
    #breakpoint()
    # https://github.com/tkipf/gcn/blob/master/gcn/utils.py
    A.setdiag(1)
    rowsum = np.array(A.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = diags(d_inv_sqrt)
    return A.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def prepare_data_for_gnn(
    dataset, use_edge_weights=True, cluster_markers_only=False, use_raw=False,
    binarize=False, remove_edges=False, remove_same_scg=True
):

    if use_raw: # use raw features instead of precomputed embeddings
        node_raw = np.hstack((dataset.node_depths, dataset.node_kmers))
        # features are already normalized
        #node_raw = (node_raw - node_raw.mean(axis=0, keepdims=True)) / node_raw.std(axis=0, keepdims=True)
        X = node_raw
    else:
        node_features = (dataset.node_embs - dataset.node_embs.mean(axis=0, keepdims=True)) / dataset.node_embs.std(
            axis=0, keepdims=True
        )
        X = node_features

    depth = 2
    # adjacency_matrix_sparse, edge_features = filter_graph_with_markers(adjacency_matrix_sparse, node_names, contig_genes, edge_features, depth=depth) 
    cluster_mask = get_cluster_mask(cluster_markers_only, dataset)
    #connected_marker_nodes = set(range(len(dataset.node_names)))
    
    adj_matrix = dataset.adj_matrix.copy()
    edge_weights = dataset.edge_weights.copy()
    if binarize:
        # both dataset.adj_matrix and dataset.edge_weights
        #breakpoint()
        percentile = 90
        threshold = np.percentile(dataset.edge_weights, percentile)
        print(f"using this threshold ({percentile} percentile) {threshold} on adj matrix with {len(dataset.adj_matrix.row)} edges")
        #dataset.adj_matrix = dataset.adj_matrix
        adj_matrix.data[adj_matrix.data < threshold] = 0
        adj_matrix.data[adj_matrix.data >= threshold] = 1
        adj_matrix.eliminate_zeros()
        edge_weights = adj_matrix.data 
        #dataset.edge_weights = np.ones(len(dataset.adj_matrix.row))
        print(f"reduce matrix to {len(edge_weights)} edges")
    
    if remove_same_scg:
        edges_with_same_scgs = dataset.get_edges_with_same_scgs()
        for x in edges_with_same_scgs:
                adj_matrix.data[x] = 0
                edge_weights[x] = 0
        adj_matrix.eliminate_zeros()
        print(f"deleted {len(edges_with_same_scgs)} edges with same SCGs")
    
    if remove_edges:
        # create self loops only sparse adj matrix
        n = len(dataset.node_names)
        adj_matrix = coo_matrix((np.ones(n), (np.array(range(n)), np.array(range(n)))), shape=(n,n))
        edge_weights = np.ones(len(adj_matrix.row))
        print(f"reduce matrix to {len(edge_weights)} edges")
    
    # gcn transform
    adj_norm = normalize_adj_sparse(adj_matrix)
    
    if use_edge_weights:
        #edge_features = (dataset.edge_weights - dataset.edge_weights.min()) / (
        #    dataset.edge_weights.max() - dataset.edge_weights.min()
        #)
        edge_features = edge_weights / edge_weights.max()
        # multiply normalized values by edge weights
        old_rows, old_cols = adj_matrix.row, adj_matrix.col
        old_idx_to_edge_idx = {(r, c): i for i, (r, c) in enumerate(zip(old_rows, old_cols))}
        old_values = adj_norm.data.astype(np.float32)
        new_values = []
        for i, j, ov in zip(adj_norm.row, adj_norm.col, old_values):
            if i == j:
                new_values.append(1.0)
            else:
                try:
                    eidx = old_idx_to_edge_idx[(i, j)]
                    new_values.append(ov * edge_features[eidx])
                except:
                    new_values.append(ov)
        new_values = np.array(new_values).astype(np.float32)
    
    else:
        adj_norm.data = np.ones(len(adj_norm.row))
        new_values = adj_norm.data.astype(np.float32)
    
    # convert to tf.SparseTensor
    adj = tf.SparseTensor(
        indices=np.array([adj_norm.row, adj_norm.col]).T, values=new_values, dense_shape=adj_norm.shape
    )
    adj = tf.sparse.reorder(adj)

    # neg_pair_idx = None
    pos_pair_idx = None
    print("**** Num of edges:", adj.indices.shape[0])
    return X, adj, cluster_mask, dataset.neg_pairs_idx, pos_pair_idx

def run_model_vaegnn(dataset, args, logger, nrun, target_metric, plot=False, use_last_batch=True):
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
    use_gnn = args.layers_gnn > 0

    with mlflow.start_run(run_name=args.assembly.split("/")[-1] + "-" + args.outname):
        mlflow.log_params(vars(args))
        
        # pick one of the LAF models
        gmodel_type = name_to_model[gname.split("_")[0].upper()] 
        clustering = args.clusteringalgo
        k = args.kclusters
        
        use_edge_weights = False
        cluster_markers_only = args.quick
        decay = 0.5 ** (2.0 / epochs)
        concat_features = args.concat_features
        use_ae = True
        args.rawfeatures = True

        # setup logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(args.outdir, 'logs/' + args.outname + current_time + '/train')
        summary_writer = tf.summary.create_file_writer(train_log_dir)
        tb_handler = TensorboardLogger(summary_writer, runname=args.outname + current_time)
        logger.addHandler(tb_handler)

        X, adj, cluster_mask, neg_pair_idx, pos_pair_idx = prepare_data_for_gnn(
            dataset, use_edge_weights, cluster_markers_only, use_raw=args.rawfeatures,
            binarize=args.binarize, remove_edges=args.noedges)

        if nrun == 0:
            print("logging to tensorboard")
            #tf.summary.trace_on(graph=True)
            logger.info("******* Running model: {} **********".format(gname))
            logger.info("***** using edge weights: {} ******".format(use_edge_weights))
            logger.info("***** concat features: {} *****".format(concat_features))
            logger.info("***** cluster markers only: {} *****".format(cluster_markers_only))
            logger.info("***** threshold adj matrix: {} *****".format(args.binarize))
            logger.info("***** self edges only: {} *****".format(args.noedges))
            logger.info("***** use gnn: {} *****".format(use_gnn))
            logger.info("***** Using raw kmer+abund features: {}".format(args.rawfeatures))
            logger.info("***** SCG neg pairs: {}".format(neg_pair_idx.shape))
            logger.info("***** input features dimension: {}".format(X[cluster_mask].shape))
        tf.config.run_functions_eagerly(True)
        
        # pre train clustering
        if not args.skip_preclustering and nrun == 0:
            cluster_labels, stats, _, hq_bins = compute_clusters_and_stats(
                        X[cluster_mask], node_names[cluster_mask],
                        dataset, clustering=clustering, k=k, tsne=args.tsne,
                        amber=(args.labels is not None and "amber" in args.labels),
                        #cuda=args.cuda,
                    )
            #mlflow.log_metrics(stats, step=0)
            if args.noise and hasattr(dataset, "true_adj_matrix"):
                # eval edge acc if true adj matrix exists
                # full adj matrix may be too big to compute, use only indices from true_adj
                #breakpoint()
                #row_embs = tf.gather(indices=dataset.true_adj_matrix.col, params=X)
                #col_embs = tf.gather(indices=dataset.true_adj_matrix.row, params=X)
                #positive_pairwise = tf.sigmoid(tf.reduce_sum(tf.math.multiply(row_embs, col_embs), axis=1))
                new_adj = tf.sigmoid(X @ X.transpose())
                predicted_edges = set([tuple(v) for v in tf.where(new_adj>0.5).numpy()])
                true_edges = set(zip(dataset.true_adj_matrix.col, dataset.true_adj_matrix.row))
                correct_edges = predicted_edges & true_edges
                print(len(correct_edges), len(true_edges), len(predicted_edges))
            logger.info(f">>> Pre train stats: {str(stats)}")
        else:
            stats = {"epoch":0, target_metric: 0, "hq": 0 }
            cluster_labels = []
        
        pname = ""

        #plot edges vs initial embs
        if plot:
            id_to_scg = {i: set(dataset.contig_markers[node_name].keys()) for i, node_name in enumerate(dataset.node_names)}
            plot_edges_sim(X, dataset.adj_matrix, id_to_scg, f"{args.outdir}/{args.outname}_pretrain_")

        # initialize variables
        scores = [stats]
        all_cluster_labels = []
        all_cluster_labels.append(cluster_labels)
        losses = {"total": [], "ae": [], "gnn": [], "scg": []}
        X = X.astype(np.float32)
        features = tf.constant(X)
        input_dim_gnn = output_dim_vae #+ dataset.node_depths.shape[1]

        logger.info(f"*** Model input dim {X.shape[1]}, GNN input dim {input_dim_gnn}")
        logger.info(f"use_ae: {use_ae}, run AE only: {args.ae_only} output clustering dim {output_dim_gnn}")

        gnn_model = gmodel_type(
            features_shape=features.shape,
            input_dim=input_dim_gnn,
            labels=None,
            adj=adj,
            embsize=output_dim_gnn,
            hidden_units=hidden_gnn,
            layers=nlayers_gnn,
            conv_last=False,
        )  # , use_bn=True, use_vae=False)

        encoder = VAEEncoder(dataset.node_depths.shape[1], dataset.node_kmers.shape[1],
                             hidden_vae, zdim=output_dim_vae, dropout=args.dropout_vae)
        decoder = VAEDecoder(dataset.node_depths.shape[1], dataset.node_kmers.shape[1],
                             hidden_vae, zdim=output_dim_vae, dropout=args.dropout_vae)

        if args.classify:
            classifier = LabelClassifier(len(dataset.labels), zdim=encoder.zdim)
        else:
            classifier = None
        trainer = TH(
            features,
            gnn_model=gnn_model,
            lr=lr_gnn,
            all_different_idx=neg_pair_idx,
            all_same_idx=pos_pair_idx,
            ae_encoder=encoder,
            ae_decoder=decoder,
            classifier=classifier,
            latentdim=output_dim_gnn,
            gnn_weight=float(args.gnn_alpha),
            ae_weight=float(args.ae_alpha),
            scg_weight=float(args.scg_alpha),
            num_negatives=args.negatives,
            kmers_dim=dataset.node_kmers.shape[1],
            abundance_dim=dataset.node_depths.shape[1],
            use_gnn=use_gnn,
            use_noise=args.noise
        )

        # create eval split
        if args.eval_split == 0:
            train_idx = np.arange(len(features))
            eval_idx = []
        else:
            train_idx = np.array(random.sample(list(range(len(features))), int(len(features)*(1-args.eval_split))))
            eval_idx = np.array([x for x in np.arange(len(features)) if x not in train_idx])
            logging.info(f"**** using {len(train_idx)} for training and {len(eval_idx)} for eval")
        edges_idx = np.arange(gnn_model.adj.indices.shape[0])
        features = np.array(features)
        pbar_epoch = tqdm(range(epochs), disable=args.quiet, position=0)
        scores = [stats]
        best_embs, best_vae_embs, best_model, best_score, best_epoch = None, None, None, 0, 0
        
        # increasing batch size
        batch_size = args.batchsize
        if batch_size == 0:
            batch_size = len(train_idx)
        logger.info("**** initial batch size: {} ****".format(batch_size))
        mlflow.log_metric("cur_batch_size", batch_size, 0)
        batch_steps = [25, 75, 150, 300]
        batch_steps = [x for i, x in enumerate(batch_steps) if (2 ** (i+1))*batch_size < len(edges_idx)]
        logger.info("**** epoch batch size doubles: {} ****".format(str(batch_steps)))
        
        vae_losses = []
        gnn_losses = []
        step = 0
        for e in pbar_epoch:
            #vae_epoch_losses = {"kld_loss": [], "vae_loss": [], "kmer_loss": [], "ab_loss": []}
            recon_loss = 0

            # train VAE in batches
            if e in batch_steps:
                print(f'Increasing batch size from {batch_size:d} to {batch_size*2:d}')
                batch_size = batch_size * 2
                mlflow.log_metric("cur_batch_size", batch_size, e )

            ############################################################

            # train model ################################################
            #np.random.shuffle(edges_idx)

            #graph_batch_size = 256
            #graph_batch_size = len(edges_idx)
            n_batches = len(edges_idx)//batch_size
            if use_last_batch and n_batches < len(edges_idx)/batch_size:
                n_batches += 1 # add final batch
            pbar_gnnbatch = tqdm(range(n_batches), disable=(args.quiet or batch_size == len(edges_idx) or n_batches < 100), position=1, ascii=' =')
            #for b in pbar_gnnbatch:
            for b in pbar_gnnbatch:
                #edges_batch = edges_idx[b*batch_size:(b+1)*batch_size]
                nodes_idx, edges_batch = train_idx[b*batch_size:(b+1)*batch_size], None
                nodes_batch = None
                losses = trainer.train_unsupervised(edges_idx=edges_batch,
                                                        nodes_idx=nodes_batch,
                                                        vae=True)
                total_loss, gnn_losses, ae_losses = losses
                #pos_loss, neg_loss, diff_loss, gnn_loss = gnn_losses
            epoch_metrics = {"Total": float(total_loss.numpy()), "gnn": gnn_losses["gnn_loss"].numpy(),
                                                "SCG": gnn_losses["scg_loss"].numpy(),
                                                #'GNN  LR': float(trainer.opt._decayed_lr(float)),
                                                "pos": gnn_losses["pos_loss"].numpy(),
                                                "neg": gnn_losses["neg_loss"].numpy(),
                                                "kld": ae_losses["kld"].numpy(),
                                                "vae": ae_losses["vae_loss"].numpy(),
                                                "kmer": ae_losses["kmer_loss"].numpy(),
                                                "ab": ae_losses["ab_loss"].numpy()}
                                                #"logvar": ae_losses["mean_logvar"],
                                                #"grad_norm": ae_losses["grad_norm"],
                                                #"grad_norm_clip": ae_losses["grad_norm_clip"]}
            #print(float(gnn_trainer.opt._decayed_lr(float)))
            log_to_tensorboard(summary_writer, epoch_metrics, step)
            mlflow.log_metrics(epoch_metrics, step=e)
            #gnn_loss = gnn_loss.numpy()
            #diff_loss = diff_loss.numpy()
            ##############################################################


            # eval loss:TODO #######################################################

            gpu_mem_alloc = tf.config.experimental.get_memory_info('GPU:0')["peak"] / 1000000 if args.cuda else 0
            #gpu_mem_alloc = tf.config.experimental.get_memory_usage('GPU:0') / 1000000 if args.cuda else 0
            # eval checkpoint ##############################################################
            if (e + 1) % RESULT_EVERY == 0 and e > args.evalskip:
                evalstarttime = datetime.datetime.now()
                #gnn_input_features = tf.concat((features[:,:dataset.node_depths.shape[1]],
                #                                gnn_trainer.encoder(features)[0]), axis=1)
                gnn_input_features = trainer.encoder(features)[0]
                logger.debug("encoder output " + str(gnn_input_features[0][:5].numpy()))
                if use_gnn:
                    node_new_features = trainer.gnn_model(gnn_input_features, None, training=False)
                    node_new_features = node_new_features.numpy()
                    if concat_features:
                        node_new_features = tf.concat([gnn_input_features, node_new_features], axis=1).numpy()
                else:
                    node_new_features = gnn_input_features.numpy()
                if args.noise:
                    mlflow.log_metrics({"avg positive noise": trainer.positive_noises.numpy().mean(),
                                                   "avg scg noise": trainer.scg_noises.numpy().mean()}, step=e)
                    # get topk edge noises
                    #breakpoint()
                    topk_indices = tf.math.top_k(tf.math.abs(trainer.positive_noises[:, 0]), k=10).indices
                    #breakpoint()
                    logger.debug("            src (label) dst (label) observed predicted noise")
                    for i in topk_indices:
                        logger.debug("{} ({}) {} ({}) {:.4f} {:.4f} {}".format(dataset.node_names[gnn_model.adj.indices[i][0]],
                                     dataset.node_to_label[dataset.node_names[gnn_model.adj.indices[i][0]]],
                                     dataset.node_names[gnn_model.adj.indices[i][1]], 
                                     dataset.node_to_label[dataset.node_names[gnn_model.adj.indices[i][1]]],
                                     gnn_model.adj.values[i].numpy(),
                                     tf.reduce_sum(tf.math.multiply(tf.nn.l2_normalize(node_new_features[gnn_model.adj.indices[i][0]]),
                                                                    tf.nn.l2_normalize(node_new_features[gnn_model.adj.indices[i][1]]))).numpy(),
                                     trainer.positive_noises[i].numpy()))


                if args.noise and hasattr(dataset, "true_adj_matrix"):
                    # eval edge acc if true adj matrix exists
                    # full adj matrix may be too big to compute, use only indices from true_adj
                    breakpoint()
                    #row_embs = tf.gather(indices=dataset.true_adj_matrix.col, params=node_new_features)
                    #col_embs = tf.gather(indices=dataset.true_adj_matrix.row, params=node_new_features)
                    #positive_pairwise = tf.sigmoid(tf.reduce_sum(tf.math.multiply(row_embs, col_embs), axis=1))
                    new_adj = tf.sigmoid(node_new_features @ node_new_features.transpose())
                    predicted_edges = set([tuple(v) for v in tf.where(new_adj>0.5).numpy()])
                    true_edges = set(zip(dataset.true_adj_matrix.col, dataset.true_adj_matrix.row))
                    correct_edges = predicted_edges & true_edges
                    print(len(correct_edges), len(true_edges), len(predicted_edges))
                    #correct_edges = dataset.true_adj_matrix.indices & 
                    #pass
                    
                weights = (trainer.encoder.get_weights(), trainer.gnn_model.get_weights())
                best_score, best_embs, best_epoch, scores, best_model, cluster_labels = eval_epoch(logger, summary_writer,
                                                                    node_new_features, cluster_mask, weights,
                                                                    step, args, dataset, e, scores,
                                                                    best_score, best_embs, best_epoch, best_model, target_metric)
                stats = scores[-1]
                all_cluster_labels.append(cluster_labels)
                if args.quiet:
                    logger.info(f"--- EPOCH {e:d} ---")
                    scores_string = f"HQ={stats['hq']}  Best{target_metric}={round(best_score, 3)} Best Epoch={best_epoch} Cur={round(stats.get(target_metric,0), 3)}"
                    losses_string = " ".join([f"{k}={v:.3f}" for k, v in epoch_metrics.items()])
                    logger.info(f"[{args.outname} {nlayers_gnn}l {pname}]{losses_string} {scores_string} GPU={gpu_mem_alloc:.1f}MB")
                    logger.info(str(stats))
                mlflow.log_metrics(stats, step=e)
                #print("total eval time", datetime.datetime.now() - evalstarttime)
            losses_string = " ".join([f"{k}={v:.3f}" for k, v in epoch_metrics.items() if v != 0])
            scores_string = f"HQ={stats['hq']} Best{target_metric}={round(best_score, 3)} BestEpoch={best_epoch} F1={round(stats.get(target_metric,0), 3)}"
            pbar_epoch.set_description(
                f"[{args.outname} {pname}] {losses_string} {scores_string} GPU={gpu_mem_alloc:.1f}MB"
            )

        #################################################################

        # concat with original features
        if concat_features:
            node_new_features = tf.concat([features, node_new_features], axis=1).numpy()
        if best_embs is None:
            best_embs = node_new_features
        
        cluster_labels, stats, _, _ = compute_clusters_and_stats(
            best_embs, node_names, dataset, clustering=clustering, k=k,
            amber=(args.labels is not None and "amber" in args.labels),
            cuda=args.cuda,
        )
        all_cluster_labels.append(cluster_labels)
        stats["epoch"] = e
        scores.append(stats)
        # get best stats:
        target_scores = [s[target_metric] for s in scores]
        best_idx = np.argmax(target_scores)
        mlflow.log_metrics(scores[best_idx], step=e+1)

        logger.info(f">>> best epoch all contigs: {RESULT_EVERY + (best_idx*RESULT_EVERY)} : {stats} <<<")
        logger.info(f">>> best epoch: {RESULT_EVERY + (best_idx*RESULT_EVERY)} : {scores[best_idx]} <<<")
        with open(f"{dataset.cache_dir}/{dataset.name}_best_contig2bin.tsv", "w") as f:
            f.write("@Version:0.9.0\n@SampleID:SAMPLEID\n@@SEQUENCEID\tBINID\n")
            for i in range(len(all_cluster_labels[best_idx])):
                f.write(f"{node_names[i]}\t{all_cluster_labels[best_idx][i]}\n")
     
        #plot edges vs initial embs
        #plot_edges_sim(best_vae_embs, dataset.adj_matrix, id_to_scg, "vae_")
        if plot:
            plot_edges_sim(best_embs, dataset.adj_matrix, id_to_scg, "posttrain_")
        return best_embs, scores[best_idx]