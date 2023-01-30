#!/bin/bash

#### baseline VAE lr 1e-3
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-3 --layers_gnn 0 --negatives 0  \
   --outname vae_lr1e-3_nodesbatch256 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 64 --batchtype nodes

#### baseline VAE lr 1e-4
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-4 --layers_gnn 0  \
   --outname vae_lr1e-4_nodesbatch256 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 32 --batchtype nodes --skip_preclustering


#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 1  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-4 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-4_edgesbatch256_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 32 --batchtype edges --skip_preclustering

#### VAE+GNN0+SCG
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 1  \
   --ae_alpha 1 --scg_alpha 1 --lr_gnn 1e-4 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-4_edgesbatch256_negs10_scg1 --nruns 3 --labels amber_ground_truth_species.tsv \
   --skip_preclustering

#### VAE+GNN3+SCG 55+1/175+3
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 1  \
   --ae_alpha 1 --scg_alpha 1 --lr_gnn 1e-4 --layers_gnn 3 --negatives 10  \
   --outname vaegcn_lr1e-4_edgesbatch256_negs10_noise --nruns 3 --labels amber_ground_truth_species.tsv \
    --skip_preclustering