#!/bin/bash


for ga in 0.1
do
for sa in 0.1
do

#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --batchsize 0 --gnn_alpha $ga  \
   --ae_alpha 1 --scg_alpha $sa --lr_gnn 1e-2 --layers_gnn 3 --negatives 10  \
   --outname vaegcn3_lr1e-2_ga${ga}_sa${sa}_pv50_fv_gd_bs0 --nruns 3   \
   --embsize_gnn 32 --skip_preclustering --quick --rawfeatures --concatfeatures \
   --vaepretrain 50 --decoder_input gnn

done
done
