import numpy as np
import datetime
import os
import tensorflow as tf
import random
import logging
from tqdm import tqdm
import mlflow
from scipy.sparse import coo_matrix, diags

from tensorflow.keras.optimizers import Adam, SGD

from graphmb.models import  TH, TrainHelperVAE, VAEDecoder, VAEEncoder, LabelClassifier
from graphmb.utils import set_seed
from graphmb.visualize import plot_edges_sim
from graphmb.evaluate import compute_clusters_and_stats, eval_epoch
from graphmb.utils import get_cluster_mask
from graphmb.gnn_models import name_to_model


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

def run_model_ccvae(dataset, args, logger, nrun, epochs=None, 
                    plot=False, use_last_batch=False, use_gnn=False,
                    target_metric="hq"):
    set_seed(args.seed)
    node_names = np.array(dataset.node_names)
    RESULT_EVERY = args.evalepochs
    hidden_gnn = args.hidden_gnn
    hidden_vae = args.hidden_vae
    output_dim_gnn = args.embsize_gnn
    output_dim_vae = args.embsize_vae
    epochs = args.epoch if not epochs else epochs
    lr_vae = args.lr_vae
    lr_gnn = args.lr_gnn
    nlayers_gnn = args.layers_gnn
    gname = args.model_name
    use_gnn = args.layers_gnn > 0 and use_gnn
    if not use_gnn:
        lr_gnn = lr_vae
    with mlflow.start_run(run_name=args.assembly.split("/")[-1] + "-" + args.outname):
        mlflow.log_params(vars(args))
        
        # pick one of the LAF models
        gmodel_type = name_to_model[gname.split("_")[0].upper()] 
        clustering = args.clusteringalgo
        k = args.kclusters
        
        use_edge_weights = True
        cluster_markers_only = args.quick
        use_ae = True

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        X, adj, cluster_mask, neg_pair_idx, pos_pair_idx = prepare_data_for_gnn(
            dataset, use_edge_weights, cluster_markers_only, use_raw=True,
            binarize=args.binarize, remove_edges=args.noedges)

        if nrun == 0:
            logger.info("******* Running model: CCVAE {}**********".format(gname if use_gnn else ""))
            logger.info("***** using edge weights: {} ******".format(use_edge_weights))
            logger.info("***** cluster markers only: {} *****".format(cluster_markers_only))
            logger.info("***** self edges only: {} *****".format(args.noedges))
            logger.info("***** Using raw kmer+abund features: {}".format(True))
            logger.info("***** SCG neg pairs: {}".format(neg_pair_idx.shape))
            logger.info("***** input features dimension: {}".format(X[cluster_mask].shape))
        tf.config.run_functions_eagerly(True)
        
        # pre train clustering
        if not args.skip_preclustering and nrun == 0:
            cluster_labels, stats, _, hq_bins = compute_clusters_and_stats(
                        X[cluster_mask], node_names[cluster_mask],
                        dataset, clustering=clustering, k=k, tsne=args.tsne,
                        amber=(args.labels is not None and "amber" in args.labels),
                        cuda=args.cuda,
                    )
            #mlflow.log_metrics(stats, step=0)
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
        if args.graph_alpha > 0:
            clustering_dim = output_dim_gnn
        elif not args.concatfeatures:
            clustering_dim = output_dim_vae
        else:
            clustering_dim = output_dim_vae + output_dim_gnn
        logger.info(f"*** Model input dim {X.shape[1]}, GNN input dim {input_dim_gnn}")
        logger.info(f"use_ae: {use_ae}, run AE only: {args.ae_only} output clustering dim {clustering_dim}")

        gnn_model = gmodel_type(
            features_shape=features.shape,
            input_dim=input_dim_gnn,
            labels=None,
            adj=adj,
            embsize=output_dim_gnn,
            hidden_units=hidden_gnn,
            layers=nlayers_gnn,
            conv_last=False,
            dropout=args.dropout_gnn
        )  # , use_bn=True, use_vae=False)

        encoder = VAEEncoder(dataset.node_depths.shape[1], dataset.node_kmers.shape[1],
                             hidden_vae, zdim=output_dim_vae, dropout=args.dropout_vae,
                             layers=args.layers_vae)
        decoder = VAEDecoder(dataset.node_depths.shape[1], dataset.node_kmers.shape[1],
                             hidden_vae, zdim=output_dim_vae, dropout=args.dropout_vae,
                             layers=args.layers_vae)

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
            decoder_input=args.decoder_input,
            classifier=classifier,
            latentdim=output_dim_gnn,
            graph_weight=float(args.graph_alpha),
            ae_weight=float(args.ae_alpha),
            scg_weight=float(args.scg_alpha),
            num_negatives=args.negatives,
            kmers_dim=dataset.node_kmers.shape[1],
            abundance_dim=dataset.node_depths.shape[1],
            use_gnn=use_gnn,
            use_noise=args.noise, # not being used
            loglevel=args.loglevel,
            pretrainvae=args.vaepretrain
        )

        if args.batchtype == "auto":
            if args.ae_alpha > 0 and args.graph_alpha == 0 and args.scg_alpha == 0:
                args.batchtype = "nodes"
            else:
                args.batchtype = "edges"
        scg_idx = np.arange(len(neg_pair_idx))
        # create eval split
        if args.eval_split == 0:
            if "node" in args.batchtype:
                train_idx = np.arange(len(features))
            elif "edge" in args.batchtype:
                train_idx = np.arange(gnn_model.adj.indices.shape[0])
            eval_idx = []
        else:
            if "node" in args.batchtype:
                train_idx = np.array(random.sample(list(range(gnn_model.adj.indices.shape[0])), int(gnn_model.adj.indices.shape[0]*(1-args.eval_split))))
                eval_idx = np.array([x for x in np.arange(gnn_model.adj.indices.shape[0]) if x not in train_idx])
            elif "edge" in args.batchtype:
                train_idx = np.array(random.sample(list(range(gnn_model.adj.indices.shape[0])), int(gnn_model.adj.indices.shape[0]*(1-args.eval_split))))
                eval_idx = np.array([x for x in np.arange(gnn_model.adj.indices.shape[0]) if x not in train_idx])
            logging.info(f"**** using {gnn_model.adj.indices.shape[0]} for training and {gnn_model.adj.indices.shape[0]} for eval")
        #edges_idx = np.arange(gnn_model.adj.indices.shape[0])
        features = np.array(features)
        pbar_epoch = tqdm(range(epochs), disable=args.quiet, position=0)
        scores = [stats]
        best_embs, best_vae_embs, best_model, best_score, best_epoch = None, None, None, 0, 0
        
        # increasing batch size
        batch_size = args.batchsize
        if batch_size == 0:
            batch_size = len(train_idx)
        logger.info("**** initial {} batch size: {} ****".format(args.batchtype, batch_size))
        mlflow.log_metric("cur_batch_size", batch_size, 0)
        batch_steps = [25, 75, 150, 300]
        batch_steps = [x for i, x in enumerate(batch_steps) if (2 ** (i+1))*batch_size < len(train_idx)]
        logger.info("**** epoch batch size doubles: {} ****".format(str(batch_steps)))
        gold_labels = np.array([dataset.labels.index(dataset.node_to_label[n]) for n in dataset.node_names])
        step = 0
        scores_string = ""
        
        for e in pbar_epoch:
            #vae_epoch_losses = {"kld_loss": [], "vae_loss": [], "kmer_loss": [], "ab_loss": []}
            recon_loss = 0
            trainer.epoch = e
            #if args.vaepretrain - 1 == e:
            #    trainer.opt = Adam(learning_rate=lr_gnn/10, epsilon=1e-8)
            # train VAE in batches
            if e in batch_steps:
                print(f'Increasing {args.batchtype} batch size from {batch_size:d} to {batch_size*2:d}')
                batch_size = batch_size * 2
                mlflow.log_metric("cur_batch_size", batch_size, e )

            ############################################################
            vae_losses_epoch = {}
            gnn_losses_epoch = {}
            total_losses_epoch = []

            # train model ################################################
            np.random.shuffle(train_idx)

            #graph_batch_size = 256
            #graph_batch_size = len(edges_idx)
            n_batches = len(train_idx)//batch_size
            if use_last_batch and n_batches < len(train_idx)/batch_size:
                n_batches += 1 # add final batch
            if args.scg_alpha > 0:
                scg_batch_size = len(neg_pair_idx)//n_batches
            pbar_gnnbatch = tqdm(range(n_batches),
                                 disable=(args.quiet or batch_size == len(train_idx) or n_batches < 1000),
                                 position=1, ascii=' =')
            for b in pbar_gnnbatch:
                if "edge" in args.batchtype:
                    nodes_batch, edges_batch = None, train_idx[b*batch_size:(b+1)*batch_size]
                elif "node" in args.batchtype:
                    nodes_batch, edges_batch = train_idx[b*batch_size:(b+1)*batch_size], None
                if args.scg_alpha > 0:
                    scg_batch = scg_idx #[b*scg_batch_size:(b+1)*scg_batch_size]
                else:
                    scg_batch = None
                losses = trainer.train_unsupervised(edges_idx=edges_batch,
                                                    nodes_idx=nodes_batch,
                                                    scgs_idx=scg_batch, vae=True,
                                                    gold_labels=gold_labels)
                total_loss, gnn_losses, ae_losses = losses
                total_losses_epoch.append(total_loss.numpy())
                for l in gnn_losses: gnn_losses_epoch.setdefault(l,[]).append(gnn_losses[l])
                for l in ae_losses: vae_losses_epoch.setdefault(l,[]).append(ae_losses[l])

                #add losses to get epoch avg
            if args.scg_alpha > 0:
                scg_loss = trainer.train_scg_only()
                scg_loss = scg_loss.numpy()
            else:
                scg_loss = 0
            epoch_metrics = {"Total": np.average(total_losses_epoch),
                             "gnn": np.average(gnn_losses_epoch["gnn_loss"]),
                             #"SCG": np.average(gnn_losses_epoch["scg_loss"]),
                             "SCG": scg_loss,
                             "pred": np.average(gnn_losses_epoch["pred_loss"]),
                            #'GNN  LR': float(trainer.opt._decayed_lr(float)),
                            "pos": np.average(gnn_losses_epoch["pos"]),
                            "neg": np.average(gnn_losses_epoch["neg"]),
                            "kld": np.average(vae_losses_epoch["kld"]),
                            "vae": np.average(vae_losses_epoch["vae_loss"]),
                            "kmer": np.average(vae_losses_epoch["kmer_loss"]),
                            "ab": np.average( vae_losses_epoch["ab_loss"])}
                            #"logvar": ae_losses["mean_logvar"],
                            #"grad_norm": ae_losses["grad_norm"],
                            #"grad_norm_clip": ae_losses["grad_norm_clip"]}
            #print(float(gnn_trainer.opt._decayed_lr(float)))
            mlflow.log_metrics(epoch_metrics, step=e)
            ##############################################################


            # eval loss:TODO #######################################################

            gpu_mem_alloc = tf.config.experimental.get_memory_info('GPU:0')["peak"] / 1000000 if args.cuda else 0
            #gpu_mem_alloc = tf.config.experimental.get_memory_usage('GPU:0') / 1000000 if args.cuda else 0
            # eval checkpoint ##############################################################
            if (e + 1) % RESULT_EVERY == 0 and e > args.evalskip and target_metric != "noeval":
                evalstarttime = datetime.datetime.now()
               
                gnn_input_features = trainer.encoder(features)[0]
                #trainer.features = gnn_input_features
                logger.debug("encoder output " + str(gnn_input_features[0][:5].numpy()))
                if use_gnn: # and e > args.vaepretrain:
                    node_new_features = trainer.gnn_model(gnn_input_features, None, training=False)
                    
                    if args.concatfeatures:
                        node_new_features = tf.concat((gnn_input_features,
                                                        node_new_features), axis=1).numpy()
                    else:
                        node_new_features = node_new_features.numpy()
                        #node_new_features = gnn_input_features.numpy()
                else:
                    node_new_features = gnn_input_features.numpy()

                    
                weights = (trainer.encoder.get_weights(), trainer.gnn_model.get_weights())
                eval_output = eval_epoch(node_new_features, cluster_mask, weights,
                                         args, dataset, e, scores, best_score, best_embs,
                                         best_epoch, best_model, target_metric)
                best_score, best_embs, best_epoch, scores, best_model, cluster_labels = eval_output
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
            scores_string = f"Best{target_metric}={round(best_score, 3)} BestEpoch={best_epoch} Cur{target_metric}={round(stats.get(target_metric,0), 3)}"
            pbar_epoch.set_description(
                f"[{args.outname} {pname}] {losses_string} {scores_string} GPU={gpu_mem_alloc:.1f}MB"
            )

        #################################################################
        if best_embs is None and target_metric != "noeval":
            best_embs = node_new_features
        else:
            node_new_features = trainer.encoder(features)[0]
        if tf.is_tensor(node_new_features):
            node_new_features = node_new_features.numpy()
        cluster_labels, stats, _, _ = compute_clusters_and_stats(
            node_new_features, node_names, dataset, clustering=clustering, k=k,
            amber=(args.labels is not None and "amber" in args.labels),
            cuda=args.cuda,
        )
        
        all_cluster_labels.append(cluster_labels)
        stats["epoch"] = e
        scores.append(stats)
        # get best stats:
        if target_metric != "noeval":
            # get best stats:
            target_scores = [s[target_metric] for s in scores]
            best_idx = np.argmax(target_scores)
        else:
            best_embs = node_new_features
            best_idx = -1
        mlflow.log_metrics(scores[best_idx], step=e+1)
        logger.info(f">>> Last epoch: {RESULT_EVERY + (best_idx*RESULT_EVERY)} : {stats} <<<")
        logger.info(f">>> best epoch: {RESULT_EVERY + (best_idx*RESULT_EVERY)} : {scores[best_idx]} <<<")
        with open(f"{args.outdir}/{args.outname}_{nrun}_best_contig2bin.tsv", "w") as f:
            f.write("@Version:0.9.0\n@SampleID:SAMPLEID\n@@SEQUENCEID\tBINID\n")
            for i in range(len(all_cluster_labels[best_idx])):
                f.write(f"{node_names[i]}\t{all_cluster_labels[best_idx][i]}\n")
     
        #plot edges vs initial embs
        #plot_edges_sim(best_vae_embs, dataset.adj_matrix, id_to_scg, "vae_")
        if plot:
            plot_edges_sim(best_embs, dataset.adj_matrix, id_to_scg, "posttrain_")
        return best_embs, scores[best_idx]