#!/bin/bash

#### baseline VAE lr 1e-3
#python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#    --epoch 1000  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0  \
#   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-3 --layers_gnn 0 --negatives 0  \
#   --outname vae_lr1e-3_nodesbatch256 --nruns 3 --labels amber_ground_truth_species.tsv \
#   --embsize_gnn 32 --batchtype nodes



#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --batchsize 256 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-3 --layers_gnn 1 --negatives 10  \
   --outname vaegcn1_lr1e-3_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 64  --skip_preclustering --quick

python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --batchsize 256 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-3 --layers_gnn 2 --negatives 10  \
   --outname vaegcn2_lr1e-3_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 64  --skip_preclustering --quick

python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --batchsize 256 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-3 --layers_gnn 2 --negatives 10  \
   --outname vaegcn2_lr1e-3_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 64  --skip_preclustering --quick