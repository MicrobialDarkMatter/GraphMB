#!/bin/bash


for ga in 0 0.1 0.2 0.3 0.5 1
do
for sa in 0 0.1 0.2 0.3 0.5 1
do

#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --batchsize 512 --gnn_alpha $ga  \
   --ae_alpha 1 --scg_alpha $sa --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-2_e512_negs10_ga${ga}_sa${sa} --nruns 3   \
   --embsize_gnn 64 --skip_preclustering --quick --rawfeatures

done
done
