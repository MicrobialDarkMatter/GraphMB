#!/bin/bash




#### VAE+GNN0
#python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#    --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
#   --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  \
#   --outname vae1gcn --nruns 3 --labels amber_ground_truth_species.tsv \
#   --embsize_gnn 64 --skip_preclustering --quick --layers_vae 1


# python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
#    --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  \
#    --outname vae2gcn --nruns 3 --labels amber_ground_truth_species.tsv \
#    --embsize_gnn 64 --skip_preclustering --quick --layers_vae 2


# python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
#    --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  \
#    --outname vae3gcn --nruns 3 --labels amber_ground_truth_species.tsv \
#    --embsize_gnn 64 --skip_preclustering --quick --layers_vae 3

# python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
#    --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  \
#    --outname vae3gcn --nruns 4 --labels amber_ground_truth_species.tsv \
#    --embsize_gnn 64 --skip_preclustering --quick --layers_vae 4

# python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
#    --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  \
#    --outname vae2gcn_vae32 --nruns 3 --labels amber_ground_truth_species.tsv \
#    --embsize_gnn 64 --skip_preclustering --quick --layers_vae 2 --hidden_vae 32

# python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
#    --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  \
#    --outname vae2gcn_vae64 --nruns 3 --labels amber_ground_truth_species.tsv \
#    --embsize_gnn 64 --skip_preclustering --quick --layers_vae 2 --hidden_vae 64

#    python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
#    --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  \
#    --outname vae2gcn_vae128 --nruns 3 --labels amber_ground_truth_species.tsv \
#    --embsize_gnn 64 --skip_preclustering --quick --layers_vae 2 --hidden_vae 128

# python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
#    --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  \
#    --outname vae2gcn_vae256 --nruns 3 --labels amber_ground_truth_species.tsv \
#    --embsize_gnn 64 --skip_preclustering --quick --layers_vae 2 --hidden_vae 256

# python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
#    --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  \
#    --outname vae2gcn_vae512 --nruns 3 --labels amber_ground_truth_species.tsv \
#    --embsize_gnn 64 --skip_preclustering --quick --layers_vae 2 --hidden_vae 512

# python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
#     --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
#    --ae_alpha 1 --scg_alpha 0.1 --lr_gnn 1e-3 --layers_gnn 0 --negatives 10  \
#    --outname vae2gcn_vae1024 --nruns 3 --labels amber_ground_truth_species.tsv \
#    --embsize_gnn 64 --skip_preclustering --quick --layers_vae 2 --hidden_vae 1024

python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0.1 --layers_gnn 0 --negatives 10  \
   --outname vae2gcn_emb16 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_vae 16 --skip_preclustering --quick

python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0.1 --layers_gnn 0 --negatives 10  \
   --outname vae2gcn_emb32 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_vae 32 --skip_preclustering --quick

python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0.1 --layers_gnn 0 --negatives 10  \
   --outname vae2gcn_emb64 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_vae 64 --skip_preclustering --quick

python src/graphmb/main.py --cuda --assembly ../data/aale/ --outdir results/aale/ --evalskip 100 \
    --epoch 1000  --model gcn_ae   --batchsize 256 --gnn_alpha 0.1  \
   --ae_alpha 1 --scg_alpha 0.1 --layers_gnn 0 --negatives 10  \
   --outname vae2gcn_emb128 --nruns 3 --labels amber_ground_truth_species.tsv \
   --embsize_vae 128 --skip_preclustering --quick
