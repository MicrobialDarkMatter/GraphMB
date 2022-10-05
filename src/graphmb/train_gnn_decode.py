import numpy as np
import datetime
import os
import tensorflow as tf
import random
import logging
from tqdm import tqdm
import mlflow

from graphmb.models import  TH, TrainHelperVAE, VAEDecoder, VAEEncoder, GVAE
from graph_functions import set_seed, run_tsne, plot_embs, plot_edges_sim
from graphmb.evaluate import calculate_overall_prf
from vaegbin import name_to_model, TensorboardLogger, prepare_data_for_gnn, compute_clusters_and_stats, log_to_tensorboard, eval_epoch

def run_model_gnn_recon(dataset, args, logger, nrun):
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
    k = args.kclusters
    use_edge_weights = True
    use_disconnected = not args.quick
    cluster_markers_only = args.quick
    decay = 0.5 ** (2.0 / epochs)
    concat_features = args.concat_features
    use_ae = gname.endswith("_ae") or args.ae_only or gname == "vae"

    with mlflow.start_run(run_name=args.assembly.split("/")[-1] + "-" + args.outname):
        mlflow.log_params(vars(args))
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
                        dataset, clustering=clustering, k=k, tsne=args.tsne, 
                        amber=(args.labels is not None and "amber" in args.labels),
                        #cuda=args.cuda,
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
        input_dim_gnn = X.shape[1]

        logger.info(f"*** Model input dim {X.shape[1]}, GNN input dim {input_dim_gnn}, use_ae: {use_ae}, run AE only: {args.ae_only}")
        
        S = []
        logger.info(f"*** output clustering dim {output_dim_gnn}")
        
        model = GVAE(dataset.node_depths.shape[1], dataset.node_kmers.shape[1],
                     X.shape[0], hidden_vae, zdim=output_dim_gnn,
                     dropout=args.dropout_vae, layers=nlayers_gnn)
        model.adj = adj
        th = TH(
            features,
            gnn_model=model,
            lr=lr_gnn,
            all_different_idx=neg_pair_idx,
            all_same_idx=pos_pair_idx,
            ae_encoder=None,
            ae_decoder=None,
            latentdim=output_dim_gnn,
            gnn_weight=float(args.gnn_alpha),
            ae_weight=float(args.ae_alpha),
            scg_weight=float(args.scg_alpha),
            num_negatives=args.negatives,
            decoder_input=args.decoder_input,
            kmers_dim=dataset.node_kmers.shape[1],
            abundance_dim=dataset.node_depth.shape[1],
        )
        th.adj = adj
        #model.summary()
    
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
        best_embs = None
        best_model = None
        best_hq = 0
        best_epoch = 0
        batch_size = args.batchsize
        if batch_size == 0:
            batch_size = len(train_idx)
        logger.info("**** initial batch size: {} ****".format(batch_size))
        batch_steps = [25, 75, 150, 300]
        batch_steps = [x for i, x in enumerate(batch_steps) if (2 ** (i+1))*batch_size < len(train_idx)]
        logger.info("**** epoch batch size doubles: {} ****".format(str(batch_steps)))
        step = 0
        for e in pbar_epoch:
            vae_epoch_losses = {"kld": [], "total": [], "kmer": [], "abundance": [], "scg": [], "gnn": []}
            np.random.shuffle(train_idx)
            recon_loss = 0

            # train VAE in batches
            if e in batch_steps:
                #print(f'Increasing batch size from {batch_size:d} to {batch_size*2:d}')
                batch_size = batch_size * 2
            np.random.shuffle(train_idx)
            n_batches = len(train_idx)//batch_size + 1
            pbar_vaebatch = tqdm(range(n_batches), disable=(args.quiet or batch_size == len(train_idx) or n_batches < 100), position=1, ascii=' =')
            for b in pbar_vaebatch:
                batch_idx = train_idx[b*batch_size:(b+1)*batch_size]
                #vae_losses = th_vae.train_step(X[batch_idx], summary_writer, step, vae=True)
                with summary_writer.as_default():
                    tf.summary.scalar('epoch', e, step=step)
        
                total_loss, gnn_loss, diff_loss, kmer_loss, ab_loss, kld_loss = th.train_unsupervised_decode(batch_idx)
                vae_epoch_losses["total"].append(total_loss)
                vae_epoch_losses["kmer"].append(kmer_loss)
                vae_epoch_losses["abundance"].append(ab_loss)
                vae_epoch_losses["kld"].append(kld_loss)
                vae_epoch_losses["scg"].append(diff_loss)
                vae_epoch_losses["gnn"].append(gnn_loss)
                gnn_loss = gnn_loss.numpy()
                diff_loss = diff_loss.numpy()
                pbar_vaebatch.set_description(f'E={e} L={np.mean(vae_epoch_losses["total"][-10:]):.4f}')
                step += 1  

            vae_epoch_losses = {k: np.mean(v) for k, v in vae_epoch_losses.items()}
            log_to_tensorboard(summary_writer, vae_epoch_losses, step)
            mlflow.log_metrics(vae_epoch_losses, step=step)
    
            if args.eval_split > 0:
                eval_mu, eval_logsigma = th_vae.encoder(X[eval_idx], training=False)
                eval_mse1, eval_mse2, eval_kld = th_vae.loss(X[eval_idx], eval_mu, eval_logsigma, vae=True, training=False)
                eval_loss = eval_mse1 + eval_mse2 - eval_kld
                log_to_tensorboard(summary_writer, {"eval_kmer": eval_mse2, "eval_ab": eval_mse1,
                                                    "eval_kld": eval_kld, "eval loss": eval_loss}, step)
            else:
                eval_loss, eval_mse1, eval_mse2, eval_kld = 0, 0, 0, 0


            #gpu_mem_alloc = tf.config.experimental.get_memory_info('GPU:0')["peak"] / 1000000 if args.cuda else 0
            gpu_mem_alloc = tf.config.experimental.get_memory_usage('GPU:0') / 1000000 if args.cuda else 0
            if (e + 1) % RESULT_EVERY == 0 and e > args.evalskip:
                #gnn_input_features = features
                #node_new_features = encoder(th.gnn_model(features, None))[0]
                node_new_features = th.gnn_model.encode(features, adj)
                #node_new_features = th.gnn_model(features, None)
                node_new_features = node_new_features.numpy()
                weights = th.gnn_model.get_weights()
                best_hq, best_embs, best_epoch, scores, best_model, cluster_labels = eval_epoch(logger, summary_writer, node_new_features,
                                                                    cluster_mask, weights, step, args, dataset, e, scores,
                                                                    best_hq, best_embs, best_epoch, best_model)
                if args.quiet:
                    logger.info(f"--- EPOCH {e:d} ---")
                    logger.info(f"[{gname} {nlayers_gnn}l {pname}] L={gnn_loss:.3f} D={diff_loss:.3f} R={recon_loss:.3f} HQ={scores[-1]['hq']}   BestHQ={best_hq} Best Epoch={best_epoch} Max GPU MB={gpu_mem_alloc:.1f}")
                    logger.info(str(stats))
                mlflow.log_metrics(scores[-1], step=step)

            losses_string = " ".join([f"{k}={v:.3f}" for k, v in vae_epoch_losses.items()])
            pbar_epoch.set_description(
                f"[{args.outname} {nlayers_gnn}l {pname}] {losses_string}  HQ={scores[-1]['hq']}  BestHQ={best_hq} Best Epoch={best_epoch} Max GPU MB={gpu_mem_alloc:.1f}"
            )
            total_loss = gnn_loss + diff_loss + recon_loss
            losses["gnn"].append(gnn_loss)
            losses["scg"].append(diff_loss)
            losses["ae"].append(recon_loss)
            losses["total"].append(total_loss)


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
    epoch_hqs = [s["epoch"] for s in scores]
    best_idx = np.argmax(hqs)
    mlflow.log_metrics(scores[best_idx], step=step+1)
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

    #plot edges vs initial embs
    #plot_edges_sim(best_vae_embs, dataset.adj_matrix, id_to_scg, "vae_")
    plot_edges_sim(best_embs, dataset.adj_matrix, id_to_scg, f"{args.outdir}/{args.outname}_posttrain_")
    return best_embs, scores[best_idx]
