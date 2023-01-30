#!/bin/bash
wwtp=$1


#### baseline VAE lr 1e-3
#python src/graphmb/main.py --cuda --assembly ../data/${wwtp}/ --outdir results/${wwtp}/ --evalskip 100 \
#    --epoch 1000 --model gcn_ae   --batchsize 512 --gnn_alpha 0  \
#   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-3 --layers_gnn 0 --negatives 0  \
#   --outname vae_lr1e-3_b512 --nruns 3   \
#   --embsize_gnn 32 --batchtype edges  --rawfeatures



#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/${wwtp}/ --outdir results/${wwtp}/ --evalskip 100 \
    --epoch 1000  --model gcn_ae   --batchsize 512 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-1 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-1_eb512_negs10 --nruns 3  \
   --embsize_gnn 64 --skip_preclustering --quick  --rawfeatures

#### VAE+GNN0+SCG
#python src/graphmb/main.py --cuda --assembly ../data/${wwtp}/ --outdir results/${wwtp}/ --evalskip 100 \
#    --epoch 1000  --model gcn_ae   --batchsize 512 --gnn_alpha 1  \
#   --ae_alpha 1 --scg_alpha 1 --lr_gnn 1e-4 --layers_gnn 0 --negatives 10  \
#   --outname vaegcn_lr1e-4_edgesbatch512_negs10_scg1 --nruns 3   \
#   --skip_preclustering

#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/${wwtp}/ --outdir results/${wwtp}/ --evalskip 100 \
    --epoch 1000  --model gcn_ae   --batchsize 512 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-2_eb512_negs10 --nruns 3   \
   --embsize_gnn 64 --skip_preclustering --quick  --rawfeatures

#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/${wwtp}/ --outdir results/${wwtp}/ --evalskip 100 \
    --epoch 1000  --model gcn_ae   --batchsize 512 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-3_eb512_negs10 --nruns 3   \
   --embsize_gnn 64 --skip_preclustering --quick  --rawfeatures

   #### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/${wwtp}/ --outdir results/${wwtp}/ --evalskip 100 \
    --epoch 1000  --model gcn_ae   --batchsize 512 --gnn_alpha 1  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 5e-4 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr5e-4_eb512_negs10 --nruns 3   \
   --embsize_gnn 64 --skip_preclustering --quick  --rawfeatures

   #### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/${wwtp}/ --outdir results/${wwtp}/ --evalskip 100 \
    --epoch 1000  --model gcn_ae   --batchsize 512 --gnn_alpha 1  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-4 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-4_eb512_negs10 --nruns 3   \
   --embsize_gnn 64 --skip_preclustering --quick  --rawfeatures

   #### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/${wwtp}/ --outdir results/${wwtp}/ --evalskip 100 \
    --epoch 1000  --model gcn_ae   --batchsize 512 --gnn_alpha 1  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-5 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-5_eb512_negs10 --nruns 3   \
   --embsize_gnn 64 --skip_preclustering --quick  --rawfeatures