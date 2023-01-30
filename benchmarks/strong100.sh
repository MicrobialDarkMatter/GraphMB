
#!/bin/bash

python src/graphmb/main.py --cuda --assembly ../data/strong100/ --outdir results/strong100/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-3 --layers_gnn 0 --negatives 0  \
   --outname vae_lr1e-3_bs256 --nruns 3 --labels amber_ground_truth.tsv \
   --embsize_gnn 32 --quick


python src/graphmb/main.py --cuda --assembly ../data/strong100/ --outdir results/strong100/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-4 --layers_gnn 0  \
   --outname vae_lr1e-4_bs256 --nruns 3 --labels amber_ground_truth.tsv \
   --embsize_gnn 32 --skip_preclustering  --quick


python src/graphmb/main.py --cuda --assembly ../data/strong100/ --outdir results/strong100/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-4 --layers_gnn 0 --negatives 5  \
   --outname vaegcn_lr1e-4_edgesbatch256_negs5 --nruns 3 --labels amber_ground_truth.tsv \
   --embsize_gnn 32 --batchtype edges --skip_preclustering  --quick


python src/graphmb/main.py --cuda --assembly ../data/strong100/ --outdir results/strong100/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0.3 --lr_gnn 1e-4 --layers_gnn 0 --negatives 5  \
   --outname vaegcn_lr1e-4_bs256_negs5_scg --nruns 3 --labels amber_ground_truth.tsv \
   --embsize_gnn 32  --skip_preclustering  --quick