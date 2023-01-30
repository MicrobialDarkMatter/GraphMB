#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0
dataset=$1
#source venv/bin/activate
#python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
#                   --model_name vae  --markers marker_gene_stats.tsv --epoch 500 \
#                   --nruns 1  --evalepochs 20  --outname vae --batchsize 256 \
#                   --evalskip 200 --labels amber_ground_truth_species.tsv
#mv results/$dataset/vae_best_embs.pickle ../data/$dataset/

#python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
#                   --model_name sage_lstm  --markers marker_gene_stats.tsv --epoch 500 \
#                   --nruns 1  --evalepochs 20  --outname sagelstm --skip_preclustering \
#                   --features vae_best_embs.pickle --concat_features --evalskip 10 --labels amber_ground_truth_species.tsv

#python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
#                   --model_name gcn  --markers marker_gene_stats.tsv --epoch 500 \
#                   --nruns 1  --evalepochs 20  --outname gcn \
#                   --features vae_best_embs.pickle --concat_features --evalskip 10 --labels amber_ground_truth_species.tsv


python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
                   --model_name gcn_ae  --markers marker_gene_stats.tsv --epoch 1000 \
                   --nruns 1  --evalepochs 20  --outname gcnae_nognn \
                   --layers_gnn 0 --evalskip 200 --batchsize 256 --labels amber_ground_truth_species.tsv

python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
                   --model_name gcn_ae  --markers marker_gene_stats.tsv --epoch 1000 \
                   --nruns 1  --evalepochs 20  --outname gcnae \
                   --concat_features --evalskip 200 --batchsize 256 --labels amber_ground_truth_species.tsv

## with GTDB

python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
                   --model_name vae  --markers gtdb --epoch 500 \
                   --nruns 1  --evalepochs 20  --outname vae_gtdb --batchsize 256 \
                   --evalskip 200 --labels amber_ground_truth_species.tsv
mv results/$dataset/vae_best_embs.pickle ../data/$dataset/

python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
                   --model_name sage_lstm  --markers gtdb --epoch 500 \
                   --nruns 1  --evalepochs 20  --outname sagelstm_gtdb --skip_preclustering \
                   --features vae_best_embs.pickle --concat_features --evalskip 10 --labels amber_ground_truth_species.tsv

python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
                   --model_name gcn  --markers gtdb --epoch 500 \
                   --nruns 1  --evalepochs 20  --outname gcn_gtdb \
                   --features vae_best_embs.pickle --concat_features --evalskip 10 --labels amber_ground_truth_species.tsv

python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
                   --model_name gcn_ae  --markers marker_gene_stats.tsv --epoch 1000 \
                   --nruns 1  --evalepochs 20  --outname gcnae_gtdb_nognn \
                   --layers_gnn 0 --evalskip 200 --batchsize 256 --labels amber_ground_truth_species.tsv

python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
                   --model_name gcn_ae  --markers gtdb --epoch 1000 \
                   --nruns 1  --evalepochs 20  --outname gcnae_gtdb \
                   --concat_features --evalskip 200 --batchsize 256 --labels amber_ground_truth_species.tsv