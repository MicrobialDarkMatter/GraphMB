import numpy as np
import datetime
import os
import tensorflow as tf
import random
import logging
from tqdm import tqdm
import mlflow
import mlflow.tensorflow

from graphmb.models import  TH, TrainHelperVAE, VAEDecoder, VAEEncoder
from graph_functions import set_seed, run_tsne, plot_embs, plot_edges_sim
from graphmb.evaluate import calculate_overall_prf
from vaegbin import name_to_model, TensorboardLogger, compute_clusters_and_stats, log_to_tensorboard, eval_epoch

def prepare_data_for_vae(dataset):
    # less preparation necessary than for GNN
    node_raw = np.hstack((dataset.node_depths, dataset.node_kmers))
    ab_dim = dataset.node_depths.shape[1]
    kmer_dim = dataset.node_kmers.shape[1]
    X = node_raw
    return X, ab_dim, kmer_dim

def run_model_vae(dataset, args, logger, nrun):
    set_seed(args.seed)
    mlflow.tensorflow.autolog()
    node_names = np.array(dataset.node_names)
    RESULT_EVERY = args.evalepochs
    hidden_vae = args.hidden_vae
    output_dim_vae = args.embsize_vae
    epochs = args.epoch
    lr_vae = args.lr_vae
    clustering = args.clusteringalgo
    k = args.kclusters
    with mlflow.start_run(run_name=args.assembly.split("/")[-1] + "-" + args.outname):
        mlflow.log_params(vars(args))
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(args.outdir, 'logs/' + args.outname + current_time + '/train')
        summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        tb_handler = TensorboardLogger(summary_writer, runname=args.outname + current_time)
        logger.addHandler(tb_handler)
        #tf.summary.trace_on(graph=True)
        if nrun == 0:
            print("logging to tensorboard")
            logger.info("******* Running model: VAE **********")
            logger.info("***** Using raw kmer+abund features: {}".format(args.rawfeatures))
        tf.config.run_functions_eagerly(True)
        X, ab_dim, kmer_dim = prepare_data_for_vae(dataset)
        cluster_mask = [True] * len(dataset.node_names)

        if not args.skip_preclustering and nrun == 0:
            cluster_labels, stats, _, hq_bins = compute_clusters_and_stats(
                        X[cluster_mask], node_names[cluster_mask], dataset, clustering=clustering, k=k) #cuda=args.cuda,
        else:
            stats = {"hq": 0, "epoch": 0}
        scores = [stats]
        losses = {"total": [], "ae": [], "gnn": [], "scg": []}
        all_cluster_labels = []
        X = X.astype(np.float32)
        features = tf.constant(X)
        input_dim_gnn = output_dim_vae
        if nrun == 0:
            logger.info(f"*** Model input dim {X.shape[1]}")
            logger.info(f"*** output clustering dim {output_dim_vae}")
        S = []
        
        gold_labels=np.array([dataset.labels.index(dataset.node_to_label[n]) for n in dataset.node_names])
        encoder = VAEEncoder(ab_dim, kmer_dim, hidden_vae, zdim=output_dim_vae, dropout=args.dropout_vae)
        decoder = VAEDecoder(ab_dim, kmer_dim, hidden_vae, zdim=output_dim_vae, dropout=args.dropout_vae)
        th_vae = TrainHelperVAE(encoder, decoder, learning_rate=lr_vae,
                                kld_weight=1/args.kld_alpha,
                                classification=args.classify, n_classes=len(dataset.labels))
        if args.eval_split == 0:
            train_idx = np.arange(len(features))
            eval_idx = []
        else:
            train_idx = np.array(random.sample(list(range(len(features))), int(len(features)*(1-args.eval_split))))
            eval_idx = np.array([x for x in np.arange(len(features)) if x not in train_idx])
            logging.info(f"**** using {len(train_idx)} for training and {len(eval_idx)} for eval")
        features = np.array(features)
        pbar_epoch = tqdm(range(epochs), disable=args.quiet, position=0)
        scores = []
        best_embs = None
        best_model = None
        best_hq = 0
        best_epoch = 0
        batch_size = args.batchsize
        if batch_size == 0:
            batch_size = len(train_idx)

        
        batch_steps = [25, 75, 150, 300]
        batch_steps = [x for i, x in enumerate(batch_steps) if (2 ** (i+1))*batch_size < len(train_idx)]
        if nrun == 0:
            logger.info("**** initial batch size: {} ****".format(batch_size))
            logger.info("**** epoch batch size doubles: {} ****".format(str(batch_steps)))
        vae_losses = []
        step = 0
        #mlflow.create_experiment(name=args.outname + current_time, tags={'run': nrun, 'dataset': args.dataset, 'model': 'vae'}) 
        #with mlflow.start_run():
        for e in pbar_epoch:
            vae_epoch_losses = {"kld_loss": [], "total_loss": [], "kmer_loss": [],
                               "ab_loss": [], "pred_loss": []}
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
                vae_losses = th_vae.train_step(X[batch_idx], summary_writer, step,
                                               vae=True, gold_labels=gold_labels[batch_idx])
                vae_epoch_losses["total_loss"].append(vae_losses[0])
                vae_epoch_losses["kmer_loss"].append(vae_losses[1])
                vae_epoch_losses["ab_loss"].append(vae_losses[2])
                vae_epoch_losses["kld_loss"].append(vae_losses[3])
                vae_epoch_losses["pred_loss"].append(vae_losses[4])
                vae_epoch_losses["kld_weight"] = th_vae.kld_weight
                vae_epoch_losses["kmer_weight"] = th_vae.kmer_weight
                #vae_epoch_losses["ab_weight"] = th_vae.abundance_weight
                #pbar_vaebatch.set_description(f'E={e} L={np.mean(vae_epoch_losses["total_loss"][-10:]):.4f}')  
                vae_epoch_losses_avg = {k: np.mean(v) for k, v in vae_epoch_losses.items()}
                losses_string = " ".join([f"{k}={v:.3f}" for k, v in vae_epoch_losses_avg.items()])
                pbar_vaebatch.set_description(f'E={e} {losses_string}')
                step += 1
            vae_epoch_losses = {k: np.mean(v) for k, v in vae_epoch_losses.items()}
            log_to_tensorboard(summary_writer, vae_epoch_losses, step)
            mlflow.log_metrics(vae_epoch_losses, step=step)

            if args.eval_split > 0:
                eval_mu, eval_logsigma = th_vae.encoder(X[eval_idx], training=False)
                eval_mse1, eval_mse2, eval_kld = th_vae.loss(X[eval_idx], eval_mu, eval_logsigma, vae=True, training=False)
                eval_loss = eval_mse1 + eval_mse2 - eval_kld
                log_to_tensorboard(summary_writer, {"eval loss": eval_loss, "eval kmer loss": eval_mse2,
                                                    "eval ab loss": eval_mse1, "eval kld loss": eval_kld}, step)

            else:
                eval_loss, eval_mse1, eval_mse2, eval_kld = 0, 0, 0, 0
            recon_loss = np.mean(vae_epoch_losses["total_loss"])

            with summary_writer.as_default():
                tf.summary.scalar('epoch', e, step=step)

            #gpu_mem_alloc = tf.config.experimental.get_memory_info('GPU:0')["peak"] / 1000000 if args.cuda else 0
            gpu_mem_alloc = tf.config.experimental.get_memory_usage('GPU:0') / 1000000 if args.cuda else 0
            if (e + 1) % RESULT_EVERY == 0 and e > args.evalskip:
            
                latent_features = encoder(features)[0]
                node_new_features = latent_features.numpy()
                if args.classify:
                    labels = th_vae.classifier(latent_features, training=False)

                with summary_writer.as_default():
                    tf.summary.scalar('Embs average', np.mean(node_new_features), step=step)
                    tf.summary.scalar('Embs std', np.std(node_new_features), step=step)
                mlflow.log_metrics({'Embs average': np.mean(node_new_features),
                                    'Embs std': np.std(node_new_features)}, step=step)
        
                cluster_labels, stats, _, hq_bins = compute_clusters_and_stats(
                    node_new_features[cluster_mask], node_names[cluster_mask],
                    dataset, clustering=clustering, k=k, tsne=args.tsne, use_labels=args.classify
                    #cuda=args.cuda,
                )

                stats["epoch"] = e
                scores.append(stats)
                #logger.info(str(stats))
                with summary_writer.as_default():
                    tf.summary.scalar('hq_bins',  stats["hq"], step=step)
                    tf.summary.scalar('mq_bins',  stats["mq"], step=step)
                mlflow.log_metrics({'hq_bins': stats["hq"], 'mq_bins': stats["mq"]}, step=step)
                all_cluster_labels.append(cluster_labels)
                if dataset.contig_markers is not None and stats["hq"] > best_hq:
                    best_hq = stats["hq"]
                    best_embs = node_new_features
                    best_epoch = e
                    #save_model(args, e, th, th_vae)

                elif dataset.contig_markers is None and stats["f1"] > best_hq:
                    best_hq = stats["f1"]
                    best_embs = node_new_features
                    best_epoch = e
                    #save_model(args, e, th, th_vae)
                # print('--- END ---')
                if args.quiet:
                    logger.info(f"--- EPOCH {e:d} ---")
                    logger.info(f"[VAE] R={recon_loss:.3f}  HQ={stats['hq']} BestHQ={best_hq} Best Epoch={best_epoch} Max GPU MB={gpu_mem_alloc:.1f}")
                    logger.info(str(stats))

            pbar_epoch.set_description(
                f"[VAE {losses_string}  HQ={stats['hq']}  BestHQ={best_hq} Best Epoch={best_epoch} Max GPU MB={gpu_mem_alloc:.1f}"
            )
            total_loss = recon_loss
            losses["ae"].append(recon_loss)
            losses["total"].append(total_loss)
    if best_embs is None:
        best_embs = node_new_features
    
    cluster_labels, stats, _, _ = compute_clusters_and_stats(
        best_embs[cluster_mask], node_names[cluster_mask],
        dataset, clustering=clustering, k=k, #cuda=args.cuda,
    )
    stats["epoch"] = e
    scores.append(stats)
    # get best stats:
    # if concat_features:  # use HQ
    hqs = [s["hq"] for s in scores]
    epoch_hqs = [s["epoch"] for s in scores]
    best_idx = np.argmax(hqs)
    # else:  # use F1
    #    f1s = [s["f1"] for s in scores]
    #    best_idx = np.argmax(f1s)
    # S.append(stats)
    S.append(scores[best_idx])
    logger.info(f">>> best epoch: {RESULT_EVERY + (best_idx*RESULT_EVERY)} : {scores[best_idx]} <<<")
    with open(f"{dataset.name}_vae_{clustering}{k}_results.tsv", "w") as f:
        f.write("@Version:0.9.0\n@SampleID:SAMPLEID\n@@SEQUENCEID\tBINID\n")
        for i in range(len(cluster_labels)):
            f.write(f"{node_names[i]}\t{cluster_labels[i]}\n")
    
    return best_embs, scores[best_idx]
