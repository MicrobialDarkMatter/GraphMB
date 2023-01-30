#!/bin/bash

wwtp=$1
bs=128
emb=32
lr=1e-3
quick=
#quick="--quick"
for ga in 0 0.1 0.2 0.3 0.5 1
do
for sa in 0 0.1 0.2 0.3 0.5 1
do

#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/$wwtp/ --outdir results/$wwtp/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --batchsize ${bs} --gnn_alpha $ga  \
   --ae_alpha 1 --scg_alpha $sa --lr_gnn ${lr} --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr${lr}_e${bs}_negs10_ga${ga}_sa${sa} --nruns 3   \
   --embsize_gnn ${emb} --skip_preclustering --rawfeatures ${quick}

done
done
