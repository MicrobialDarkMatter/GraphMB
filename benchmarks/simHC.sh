#!/bin/bash
# simHC experiments

python src/graphmb/main.py --cuda --assembly ../data/simHC/ --outdir results/simHC/ --assembly_name contigs.fasta --depth abundance.tsv --contignodes --evalskip 0 --epoch 100  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0 --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  --outname ae_lr1e-3 --nruns 1 --labels amber_ground_truth.tsv --embsize_gnn 32