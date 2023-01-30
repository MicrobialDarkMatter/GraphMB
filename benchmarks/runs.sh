# VAE

# multiple runs with filtered nodes (SCG and connected only)
############################################################


dataset=$2
# GCN on pre-trained AE features
export CUDA_VISIBLE_DEVICES=$1
quick=" --quick --nruns 5"
#quick=""
addname="_fixloss"

#python src/graphmb/main.py --cuda --assembly ../data/$dataset --outdir results/$dataset --model_name vae \
#             --markers marker_gene_stats.tsv --batchsize 256 --epoch 500 --lr_vae 1e-3 \
#             --nruns 5  --evalepochs 20 --outname vae_baseline 

python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
                             --model_name gcn  --markers marker_gene_stats.tsv --epoch 500 \
                             --evalepochs 20  --outname gcn_lr1e-4$addname --lr_gnn 1e-4 \
                             --features vae_baseline_best_embs.pickle --concat_features --evalskip 200 $quick

python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
                             --model_name gcn  --markers marker_gene_stats.tsv --epoch 500 \
                             --evalepochs 20  --outname gcn_lr1e-3$addname --lr_gnn 1e-3 \
                             --features vae_best_embs.pickle --concat_features --evalskip 200 $quick

# VAE+GCN model (separate losses)
python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
         --model_name gcn_ae --markers marker_gene_stats.tsv --epoch 500 \
         --evalepochs 20  --outname gcnae_lr1e-4$addname --lr_gnn 1e-4 \
         --batchsize 256 --rawfeatures --gnn_alpha 0.5 --scg_alpha 100 --concat_features \
         --evalskip 100 --skip_preclustering $quick


# GVAE model, reconloss
python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
            --model_name gcn_decode --markers marker_gene_stats.tsv --epoch 500  \
            --evalepochs 20  --outname gcndecode_lr1e-4$addname --lr_gnn 1e-4 --batchsize 256 \
            --rawfeatures --gnn_alpha 0.5 --scg_alpha 0 --evalskip 100 --skip_preclustering --layers_gnn 3 $quick



# Using only top 10% of edges (separate losses)
python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
         --model_name gcn_ae --markers marker_gene_stats.tsv --epoch 500 \
         --evalepochs 20  --outname gcnae_lr1e-4_binarize$addname --lr_gnn 1e-4 \
         --batchsize 256 --rawfeatures --gnn_alpha 0.5 --scg_alpha 100 --concat_features \
         --evalskip 100 --skip_preclustering --binarize $quick


# VAE+GCN augmented graph
python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
                   --model_name gcn_aug  --markers marker_gene_stats.tsv --epoch 500 \
                    --evalepochs 20  --outname gcnaug_lr1e-4$addname --concat_features \
                   --lr_gnn 1e-4 --rawfeatures --evalskip 100 $quick


### extra experiments

### no edges
python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
                             --model_name gcn  --markers marker_gene_stats.tsv --epoch 500 \
                             --evalepochs 20  --outname gcn_lr1e-3_noedges$addname --noedges --lr_gnn 1e-3 \
                             --features vae_best_embs.pickle --concat_features --evalskip 200 $quick
python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
         --model_name gcn_ae --markers marker_gene_stats.tsv --epoch 500 \
         --evalepochs 20  --outname gcnae_lr1e-4_noedges$addname --noedges --lr_gnn 1e-4 \
         --batchsize 256 --rawfeatures --gnn_alpha 0.5 --scg_alpha 100 --concat_features \
         --evalskip 100 --skip_preclustering $quick
# GVAE model, reconloss
python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
            --model_name gcn_decode --markers marker_gene_stats.tsv --epoch 500  \
            --evalepochs 20  --outname gcndecode_lr1e-4_noedges$addname --noedges --lr_gnn 1e-4 --batchsize 256 \
            --rawfeatures --gnn_alpha 0.5 --scg_alpha 0 --evalskip 100 --skip_preclustering --layers_gnn 3 $quick
python src/graphmb/main.py --cuda --assembly ../data/$dataset/ --outdir results/$dataset/ \
                   --model_name gcn_aug  --markers marker_gene_stats.tsv --epoch 500 \
                    --evalepochs 20  --outname gcnaug_lr1e-4_noedges$addname --noedges --concat_features \
                   --lr_gnn 1e-4 --rawfeatures --evalskip 100 $quick