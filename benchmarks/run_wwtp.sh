#!/bin/bash
wwtp=$1

#### baseline VAE lr 1e-2
#python src/graphmb/main.py --cuda --assembly ../data/$wwtp/ --outdir results/$wwtp/ --evalskip 100 \
#    --epoch 500  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0  \
#   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-2 --layers_gnn 0 --negatives 0  \
#   --outname vae_lr1e-2_nb256 --nruns 3 \
#   --embsize_gnn 64 --quick --batchtype nodes


#### baseline VAE lr 1e-2
#python src/graphmb/main.py --cuda --assembly ../data/$wwtp/ --outdir results/$wwtp/ --evalskip 100 \
#    --epoch 500  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0  \
#   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-3 --layers_gnn 0 --negatives 0  \
#   --outname vae_lr1e-3_nb256 --nruns 3 \
#   --embsize_gnn 64 --quick --batchtype nodes

#### baseline VAE lr 1e-4
#python src/graphmb/main.py --cuda --assembly ../data/$wwtp/ --outdir results/$wwtp/ --evalskip 100 \
#    --epoch 1000  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0  \
#   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-4 --layers_gnn 0  \
#   --outname vae_lr1e-4_nb256 --nruns 3 \
#   --skip_preclustering --embsize_gnn 64 --quick --batchtype nodes


#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/$wwtp/ --outdir results/$wwtp/ --evalskip 100 \
    --epoch 2000 --evalepoch 20 --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-3_eb256_negs10_gnn0.1 --nruns 3 \
    --skip_preclustering --embsize_gnn 64 --quick

#### VAE+GNN0+SCG 
python src/graphmb/main.py --cuda --assembly ../data/$wwtp/ --outdir results/$wwtp/ --evalskip 10 \
    --epoch 2000 --evalepoch 20 --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0.3 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-3_eb256_negs10_scg0.1_gnn0.3 --nruns 3 \
    --skip_preclustering --embsize_gnn 64 --quick
