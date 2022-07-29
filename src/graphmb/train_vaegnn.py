import numpy as np
import datetime
import os
import tensorflow as tf
import random
import logging
from tqdm import tqdm

from graphmb.models import  TH, TrainHelperVAE, VAEDecoder, VAEEncoder
from graph_functions import set_seed, run_tsne, plot_embs, plot_edges_sim
from graphmb.evaluate import calculate_overall_prf
from vaegbin import name_to_model, TensorboardLogger, prepare_data_for_gnn, compute_clusters_and_stats, log_to_tensorboard, eval_epoch

def run_model_vaegnn(dataset, args, logger):
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


    X, adj, cluster_mask, neg_pair_idx, pos_pair_idx, ab_dim, kmer_dim = prepare_data_for_gnn(
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
    plot_edges_sim(X, dataset.adj_matrix, id_to_scg, "pretrain_")

    scores = [stats]
    losses = {"total": [], "ae": [], "gnn": [], "scg": []}
    all_cluster_labels = []
    X = X.astype(np.float32)
    features = tf.constant(X)
    input_dim_gnn = output_dim_vae

    logger.info(f"*** Model input dim {X.shape[1]}, GNN input dim {input_dim_gnn}, use_ae: {use_ae}, run AE only: {args.ae_only}")
    
    S = []

    gnn_model = gmodel_type(
        features_shape=features.shape,
        input_dim=input_dim_gnn,
        labels=None,
        adj=adj,
        n_labels=output_dim_gnn,
        hidden_units=hidden_gnn,
        layers=nlayers_gnn,
        conv_last=False,
    )  # , use_bn=True, use_vae=False)
    logger.info(f"*** output clustering dim {output_dim_gnn}")

    encoder = VAEEncoder(ab_dim, kmer_dim, hidden_vae, zdim=output_dim_vae, dropout=args.dropout_vae)
    decoder = VAEDecoder(ab_dim, kmer_dim, hidden_vae, zdim=output_dim_vae, dropout=args.dropout_vae)
    th_vae = TrainHelperVAE(encoder, decoder, learning_rate=lr_vae, kld_weight=1/args.kld_alpha)

    th = TH(
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
        kmers_dim=kmer_dim,
        abundance_dim=ab_dim,
    )


    if not args.quiet:
        if not args.ae_only:
            gnn_model.summary()
        #if gname.endswith("_ae"):
        #    th.encoder.summary()
        #    th.decoder.summary()
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
    batch_size = args.batchsize
    if batch_size == 0:
        batch_size = len(train_idx)
        logger.info("**** initial batch size: {} ****".format(batch_size))
    batch_steps = [25, 75, 150, 300]
    batch_steps = [x for i, x in enumerate(batch_steps) if (2 ** (i+1))*batch_size < len(train_idx)]
    logger.info("**** epoch batch size doubles: {} ****".format(str(batch_steps)))
    vae_losses = []
    step = 0
    for e in pbar_epoch:
        vae_epoch_losses = {"kld": [], "total vae loss": [], "kmer": [], "ab": []}
        np.random.shuffle(train_idx)
        recon_loss = 0

        # train VAE in batches
        if e in batch_steps:
            #print(f'Increasing batch size from {batch_size:d} to {batch_size*2:d}')
            batch_size = batch_size * 2
        np.random.shuffle(train_idx)
        n_batches = len(train_idx)//batch_size
        pbar_vaebatch = tqdm(range(n_batches), disable=(args.quiet or batch_size == len(train_idx) or n_batches < 100), position=1, ascii=' =')
        for b in pbar_vaebatch:
            batch_idx = train_idx[b*batch_size:(b+1)*batch_size]
            vae_losses = th_vae.train_step(X[batch_idx], summary_writer, step, vae=True)
            vae_epoch_losses["total vae loss"].append(vae_losses[0])
            vae_epoch_losses["kmer"].append(vae_losses[1])
            vae_epoch_losses["ab"].append(vae_losses[2])
            vae_epoch_losses["kld"].append(vae_losses[3])
            pbar_vaebatch.set_description(f'E={e} L={np.mean(vae_epoch_losses["total vae loss"][-10:]):.4f}')
            step += 1
        vae_epoch_losses = {k: np.mean(v) for k, v in vae_epoch_losses.items()}
        log_to_tensorboard(summary_writer, vae_epoch_losses, step)

        if args.eval_split > 0:
            eval_mu, eval_logsigma = th_vae.encoder(X[eval_idx], training=False)
            eval_mse1, eval_mse2, eval_kld = th_vae.loss(X[eval_idx], eval_mu, eval_logsigma, vae=True, training=False)
            eval_loss = eval_mse1 + eval_mse2 - eval_kld
            log_to_tensorboard(summary_writer, {"eval_loss": eval_loss, "eval kmer loss": eval_mse2,
                                                "eval ab loss": eval_mse1, "eval kld loss": eval_kld}, step)
        else:
            eval_loss, eval_mse1, eval_mse2, eval_kld = 0, 0, 0, 0
        recon_loss = np.mean(vae_epoch_losses["total vae loss"])

        th.encoder = th_vae.encoder
            
        with summary_writer.as_default():
            tf.summary.scalar('epoch', e, step=step)
 
        total_loss, gnn_loss, diff_loss = th.train_unsupervised(train_idx)
        log_to_tensorboard(summary_writer, {"Total loss": total_loss, "gnn loss": gnn_loss, "SCG loss": diff_loss,
                                            'GNN  LR': th.opt.learning_rate}, step)
        gnn_loss = gnn_loss.numpy()
        diff_loss = diff_loss.numpy()

        #gpu_mem_alloc = tf.config.experimental.get_memory_info('GPU:0')["peak"] / 1000000 if args.cuda else 0
        gpu_mem_alloc = tf.config.experimental.get_memory_usage('GPU:0') / 1000000 if args.cuda else 0
        if (e + 1) % RESULT_EVERY == 0 and e > args.evalskip:
            if not args.ae_only:
                th.gnn_model.adj = adj
            gnn_input_features = features
            if use_ae:
                gnn_input_features = encoder(features)[0]
            if not args.ae_only:
                node_new_features = th.gnn_model(gnn_input_features, None, training=False)
                node_new_features = node_new_features.numpy()
            else:
                node_new_features = gnn_input_features.numpy()

            if concat_features:
                node_new_features = tf.concat([gnn_input_features, node_new_features], axis=1).numpy()

            best_hq, best_embs, best_epoch, scores = eval_epoch(logger, summary_writer, node_new_features,
                                                                cluster_mask, step, args, dataset, e, scores,
                                                                best_hq, best_embs, best_epoch)
            
            if args.quiet:
                logger.info(f"--- EPOCH {e:d} ---")
                logger.info(f"[{gname} {nlayers_gnn}l {pname}] L={gnn_loss:.3f} D={diff_loss:.3f} R={recon_loss:.3f} HQ={stats['hq']}   BestHQ={best_hq} Best Epoch={best_epoch} Max GPU MB={gpu_mem_alloc:.1f}")
                logger.info(str(scores[-1]))
        losses_string = " ".join([f"{k}={v:.3f}" for k, v in vae_epoch_losses.items()])
        pbar_epoch.set_description(
            f"[{gname} {nlayers_gnn}l {pname}] GNN={gnn_loss:.3f} SCG={diff_loss:.3f} {losses_string} HQ={scores[-1]['hq']}  BestHQ={best_hq} Best Epoch={best_epoch} Max GPU MB={gpu_mem_alloc:.1f}"
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

  

    # concat with original features
    if concat_features:
        node_new_features = tf.concat([features, node_new_features], axis=1).numpy()
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
    plot_edges_sim(best_embs, dataset.adj_matrix, id_to_scg, "posttrain_")
    return best_embs, scores[best_idx]