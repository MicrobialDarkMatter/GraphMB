import numpy as np
import datetime
import os
import tensorflow as tf
import random
import logging
from tqdm import tqdm
import mlflow

from graphmb.models import  TH, TrainHelperVAE, VAEDecoder, VAEEncoder
from graph_functions import set_seed, run_tsne, plot_embs, plot_edges_sim
from graphmb.evaluate import calculate_overall_prf
from vaegbin import name_to_model, TensorboardLogger, prepare_data_for_gnn, compute_clusters_and_stats, log_to_tensorboard, eval_epoch

def run_model_vaegnn(dataset, args, logger, nrun, plot=False):
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
        
        use_edge_weights = True
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
                        cuda=args.cuda,
                    )
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
            stats = {"hq": 0, "epoch":0 }
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
        vae_trainer = TrainHelperVAE(encoder, decoder, learning_rate=lr_vae,
                                     kld_weight=1/args.kld_alpha, ae_alpha=args.ae_alpha)
        gnn_trainer = TH(
            features,
            gnn_model=gnn_model,
            lr=lr_gnn,
            all_different_idx=neg_pair_idx,
            all_same_idx=pos_pair_idx,
            ae_encoder=encoder,
            ae_decoder=decoder,
            latentdim=output_dim_gnn,
            gnn_weight=float(args.gnn_alpha),
            ae_weight=float(args.ae_alpha),
            scg_weight=float(args.scg_alpha),
            num_negatives=args.negatives,
            decoder_input=args.decoder_input,
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
        
        features = np.array(features)
        pbar_epoch = tqdm(range(epochs), disable=args.quiet, position=0)
        scores = [stats]
        best_embs, best_vae_embs, best_model, best_hq, best_epoch = None, None, None, 0, 0
        
        # increasing batch size
        batch_size = args.batchsize
        if batch_size == 0:
            batch_size = len(train_idx)
        logger.info("**** initial batch size: {} ****".format(batch_size))
        mlflow.log_metric("cur_batch_size", batch_size, 0)
        batch_steps = [25, 75, 150, 300]
        batch_steps = [x for i, x in enumerate(batch_steps) if (2 ** (i+1))*batch_size < len(train_idx)]
        batch_steps = [x for i, x in enumerate(batch_steps) if (2 ** (i+1))*batch_size < len(train_idx)]
        logger.info("**** epoch batch size doubles: {} ****".format(str(batch_steps)))
        
        vae_losses = []
        gnn_losses = []
        step = 0
        for e in pbar_epoch:
            vae_epoch_losses = {"kld_loss": [], "vae_loss": [], "kmer_loss": [], "ab_loss": []}
            recon_loss = 0

            # train VAE in batches
            #if e in batch_steps:
                #print(f'Increasing batch size from {batch_size:d} to {batch_size*2:d}')
                #batch_size = batch_size * 2
                #mlflow.log_metric("cur_batch_size", batch_size, e )


            # train VAE ########
            np.random.shuffle(train_idx)
            n_batches = len(train_idx)//batch_size
            if n_batches < len(train_idx)/batch_size:
                n_batches += 1 # add final batch
            pbar_vaebatch = tqdm(range(n_batches), disable=(args.quiet or batch_size == len(train_idx) or n_batches < 100), position=1, ascii=' =')
            for b in pbar_vaebatch:
                batch_idx = train_idx[b*batch_size:(b+1)*batch_size]
                batch_vae_losses = vae_trainer.train_step(X[batch_idx], summary_writer, step, vae=True)
                vae_epoch_losses["vae_loss"].append(batch_vae_losses[0])
                vae_epoch_losses["kmer_loss"].append(batch_vae_losses[1])
                vae_epoch_losses["ab_loss"].append(batch_vae_losses[2])
                vae_epoch_losses["kld_loss"].append(batch_vae_losses[3])
                pbar_vaebatch.set_description(f'E={e} L={np.mean(vae_epoch_losses["vae_loss"][-10:]):.4f}')
                step += 1
            vae_epoch_losses = {k: np.mean(v) for k, v in vae_epoch_losses.items()}
            vae_losses.append(vae_epoch_losses["vae_loss"])
            log_to_tensorboard(summary_writer, vae_epoch_losses, step)
            mlflow.log_metrics(vae_epoch_losses, step=e)
            recon_loss = np.mean(vae_epoch_losses["vae_loss"])
            # update batch size
            if len(vae_losses) > 20 and sum(recon_loss < np.array(vae_losses[-20:-1])) < 5 and batch_size*2 < len(train_idx):
                batch_size *= 2
                print("duplicate batch size", e, batch_size, recon_loss, vae_losses[-5:])
                mlflow.log_metric("cur_batch_size", batch_size, e)
                vae_losses = []
            ############################################################

            # train GNN ################################################
            gnn_trainer.encoder = vae_trainer.encoder
            edges_idx = np.arange(gnn_model.adj.indices.shape[0])
            np.random.shuffle(edges_idx)
            #graph_batch_size = 256
            graph_batch_size = len(edges_idx)
            n_batches = len(edges_idx)//graph_batch_size
            if n_batches < len(edges_idx)/graph_batch_size:
                n_batches += 1 # add final batch
            pbar_gnnbatch = tqdm(range(n_batches), disable=(args.quiet or graph_batch_size == len(edges_idx) or n_batches < 100), position=1, ascii=' =')
            for b in pbar_gnnbatch:
                batch_idx = edges_idx[b*graph_batch_size:(b+1)*graph_batch_size]
                total_loss, gnn_loss, diff_loss, pos_loss, neg_loss = gnn_trainer.train_unsupervised(edges_idx=edges_idx, nodes_idx=train_idx)
            epoch_metrics = {"Total loss": float(total_loss), "gnn loss": float(gnn_loss),
                                                "SCG loss": float(diff_loss),
                                                #'GNN  LR': float(gnn_trainer.opt._decayed_lr(float)),
                                                "pos loss": float(pos_loss), "neg loss": float(neg_loss)}
            #print(float(gnn_trainer.opt._decayed_lr(float)))
            log_to_tensorboard(summary_writer, epoch_metrics, step)
            mlflow.log_metrics(epoch_metrics, step=e)
            gnn_loss = gnn_loss.numpy()
            diff_loss = diff_loss.numpy()
            ##############################################################


            # eval loss ########################################################
            if args.eval_split > 0:
                eval_mu, eval_logsigma = vae_trainer.encoder(X[eval_idx], training=False)
                eval_vae_losses = vae_trainer.loss(X[eval_idx], eval_mu, eval_logsigma, vae=True, training=False)
                #eval_loss = eval_mse1 + eval_mse2 - eval_kld
                eval_vae_epoch_losses = {}
                eval_vae_epoch_losses["vae_loss"] = float(eval_vae_losses[0])
                eval_vae_epoch_losses["kmer_loss"] = float(eval_vae_losses[1])
                eval_vae_epoch_losses["ab_loss"] = float(eval_vae_losses[2])
                eval_vae_epoch_losses["kld_loss"] = float(eval_vae_losses[3])
                mlflow.log_metrics(eval_vae_epoch_losses, step=e)
                eval_total_loss, eval_gnn_loss, eval_diff_loss, eval_pos_loss, \
                    eval_neg_loss = gnn_trainer.train_unsupervised(eval_idx, training=False)
                eval_metrics = {"Eval loss": float(eval_total_loss),
                                 "eval gnn loss": float(eval_gnn_loss),
                                 "eval SCG loss": float(eval_diff_loss),
                                 "eval pos loss": float(eval_pos_loss), "neg loss": float(eval_neg_loss)}
                log_to_tensorboard(summary_writer, eval_metrics, step)
                mlflow.log_metrics(eval_metrics, step=e)


            #gpu_mem_alloc = tf.config.experimental.get_memory_info('GPU:0')["peak"] / 1000000 if args.cuda else 0
            gpu_mem_alloc = tf.config.experimental.get_memory_usage('GPU:0') / 1000000 if args.cuda else 0
            # eval checkpoint ##############################################################
            if (e + 1) % RESULT_EVERY == 0 and e > args.evalskip:
                evalstarttime = datetime.datetime.now()
                #gnn_input_features = tf.concat((features[:,:dataset.node_depths.shape[1]],
                #                                gnn_trainer.encoder(features)[0]), axis=1)
                gnn_input_features = gnn_trainer.encoder(features)[0]
                logger.debug(str(gnn_input_features[0][:5].numpy()))
                if use_gnn:
                    node_new_features = gnn_trainer.gnn_model(gnn_input_features, None, training=False)
                    node_new_features = node_new_features.numpy()
                    if concat_features:
                        node_new_features = tf.concat([gnn_input_features, node_new_features], axis=1).numpy()
                else:
                    node_new_features = gnn_input_features.numpy()
                if args.noise: mlflow.log_metrics({"avg positive noise": gnn_trainer.positive_noises.numpy().mean(),
                                                   "avg scg noise": gnn_trainer.scg_noises.numpy().mean()}, step=e)
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
                    
                weights = (gnn_trainer.encoder.get_weights(), gnn_trainer.gnn_model.get_weights())
                best_hq, best_embs, best_epoch, scores, best_model, cluster_labels = eval_epoch(logger, summary_writer,
                                                                    node_new_features, cluster_mask, weights,
                                                                    step, args, dataset, e, scores,
                                                                    best_hq, best_embs, best_epoch, best_model)
                stats = scores[-1]
                all_cluster_labels.append(cluster_labels)
                if args.quiet:
                    logger.info(f"--- EPOCH {e:d} ---")
                    scores_string = f"HQ={stats['hq']}  BestHQ={best_hq} Best Epoch={best_epoch} F1={round(stats.get('f1_avg_bp',0), 3)}"
                    losses_string = " ".join([f"{k}={v:.3f}" for k, v in vae_epoch_losses.items()])
                    logger.info(f"[{args.outname} {nlayers_gnn}l {pname}] GNN={gnn_loss:.3f} SCG={diff_loss:.3f} {losses_string} {scores_string} GPU={gpu_mem_alloc:.1f}MB")
                    logger.info(str(stats))
                mlflow.log_metrics(stats, step=e)
                #print("total eval time", datetime.datetime.now() - evalstarttime)
            losses_string = " ".join([f"{k}={v:.3f}" for k, v in vae_epoch_losses.items()])
            scores_string = f"HQ={stats['hq']}  BestHQ={best_hq} Best Epoch={best_epoch} F1={round(stats.get('f1_avg_bp',0), 3)}"
            pbar_epoch.set_description(
                f"[{args.outname} {nlayers_gnn}l {pname}] GNN={gnn_loss:.3f} SCG={diff_loss:.3f} {losses_string} {scores_string} GPU={gpu_mem_alloc:.1f}MB"
            )
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
        # if concat_features:  # use HQ
        hqs = [s["hq"] for s in scores]
        #epoch_hqs = [s["epoch"] for s in scores]
        best_idx = np.argmax(hqs)
        mlflow.log_metrics(scores[best_idx], step=e+1)
        # else:  # use F1
        #    f1s = [s["f1"] for s in scores]
        #    best_idx = np.argmax(f1s)
        logger.info(f">>> best epoch all contigs: {RESULT_EVERY + (best_idx*RESULT_EVERY)} : {stats} <<<")
        logger.info(f">>> best epoch: {RESULT_EVERY + (best_idx*RESULT_EVERY)} : {scores[best_idx]} <<<")
        with open(f"{dataset.cache_dir}/{dataset.name}_best_contig2bin.tsv", "w") as f:
            f.write("@Version:0.9.0\n@SampleID:SAMPLEID\n@@SEQUENCEID\tBINID\n")
            for i in range(len(all_cluster_labels[best_idx])):
                f.write(f"{node_names[i]}\t{all_cluster_labels[best_idx][i]}\n")
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
        if plot:
            plot_edges_sim(best_embs, dataset.adj_matrix, id_to_scg, "posttrain_")
        return best_embs, scores[best_idx]