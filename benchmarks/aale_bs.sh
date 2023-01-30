#!/bin/bash

#### baseline VAE lr 1e-3
# python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae --rawfeatures --batchsize 0 --gnn_alpha 0  \
#    --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-2 --layers_gnn 0 --negatives 0  \
#    --outname vae_lr1e-2_bs0 --nruns 3 \
#    --embsize_gnn 64 --quick

# python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae --rawfeatures --batchsize 128 --gnn_alpha 0  \
#    --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-2 --layers_gnn 0 --negatives 0  \
#    --outname vae_lr1e-2_bs128 --nruns 3 \
#    --embsize_gnn 64 --quick

# python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae --rawfeatures --batchsize 512 --gnn_alpha 0  \
#    --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-2 --layers_gnn 0 --negatives 0  \
#    --outname vae_lr1e-2_bs512 --nruns 3 \
#    --embsize_gnn 64 --quick

# python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae --rawfeatures --batchsize 1024 --gnn_alpha 0  \
#    --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-2 --layers_gnn 0 --negatives 0  \
#    --outname vae_lr1e-2_bs1024 --nruns 3 \
#    --embsize_gnn 64 --quick

# python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae --rawfeatures --batchsize 10000 --gnn_alpha 0  \
#    --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-2 --layers_gnn 0 --negatives 0  \
#    --outname vae_lr1e-2_bs10k --nruns 3 \
#    --embsize_gnn 64 --quick


# ################################################

#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --batchsize 0 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-2_bs0_negs10 --nruns 3 \
   --embsize_gnn 64  --skip_preclustering --quick

#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --batchsize 128 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-2_eb128_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 64  --skip_preclustering --quick

#### VAE+GNN0+SCG
#python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#    --epoch 1000  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 1  \
#   --ae_alpha 1 --scg_alpha 1 --lr_gnn 1e-4 --layers_gnn 0 --negatives 10  \
#   --outname vaegcn_lr1e-4_edgesbatch256_negs10_scg1 --nruns 3 --labels amber_ground_truth_species.tsv \
#   --skip_preclustering

#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae  --batchsize 256 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-2_eb256_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 64 --skip_preclustering --quick

#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae  --batchsize 512 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-2_eb512_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 64 --skip_preclustering --quick

### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae  --batchsize 1024 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
   --outname vaegcn_lr1e-2_eb1024_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 64 --skip_preclustering --quick

### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
   --epoch 1000  --model gcn_ae  --batchsize 10000 --gnn_alpha 0.1  \
  --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
  --outname vaegcn_lr1e-2_eb10k_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
  --embsize_gnn 64 --skip_preclustering --quick


########################
#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --batchsize 0 --gnn_alpha 1  \
   --ae_alpha 0 --scg_alpha 1 --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
   --outname vae0gcn_lr1e-2_bs0_negs10 --nruns 3 \
   --embsize_gnn 64  --skip_preclustering --quick

#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae --batchsize 128 --gnn_alpha 1  \
   --ae_alpha 0 --scg_alpha 1 --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
   --outname vae0gcn_lr1e-2_eb128_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 64  --skip_preclustering --quick



#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae  --batchsize 256 --gnn_alpha 1  \
   --ae_alpha 0 --scg_alpha 1 --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
   --outname vae0gcn_lr1e-2_eb256_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 64 --skip_preclustering --quick

#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae  --batchsize 512 --gnn_alpha 1  \
   --ae_alpha 0 --scg_alpha 1 --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
   --outname vae0gcn_lr1e-2_eb512_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 64 --skip_preclustering --quick

#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae  --batchsize 1024 --gnn_alpha 1  \
   --ae_alpha 0 --scg_alpha 1 --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
   --outname vae0gcn_lr1e-2_eb1024_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 64 --skip_preclustering --quick

#### VAE+GNN0
python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae  --batchsize 10000 --gnn_alpha 1  \
   --ae_alpha 0 --scg_alpha 1 --lr_gnn 1e-2 --layers_gnn 0 --negatives 10  \
   --outname vae0gcn_lr1e-2_eb10k_negs10 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_gnn 64 --skip_preclustering --quick
