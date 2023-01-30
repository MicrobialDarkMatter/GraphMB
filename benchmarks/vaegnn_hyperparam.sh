set -x
#set -e
DATADIR=../data/$2/
RESULTSDIR=results/$2/
CUDA=$1
nruns="2"
epoch=500
evalsplit=0
model="gcn_ae"
evalskip=100
concatfeatures="--concat_features"
common_params="--layers_gnn 0 --batchsize 256"


# run VAE only

# CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name vae \
#              --markers marker_gene_stats.tsv --batchsize 128 --epoch 1000 --lr_vae 1e-3 \
#              --nruns 1  --evalepochs 20 --skip_preclustering --outname vae_baseline



# run GNN with saved features from VAE only


# run VAE-GNN

# VAE embsize
# CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
#                --epoch $epoch --nruns $nruns  --evalepochs 50  --outname ${model}_nognn_bs128_emb16 --embsize_vae 16 \
#                --batchsize 128  $concatfeatures  --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 0 

# CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
#                --epoch $epoch --nruns $nruns  --evalepochs 50  --outname ${model}_nognn_bs128_emb32 --embsize_vae 32 \
#                --batchsize 128  $concatfeatures  --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 0 

# CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
#                --epoch $epoch --nruns $nruns  --evalepochs 50  --outname ${model}_nognn_bs128_emb64 --embsize_vae 64 \
#                --batchsize 128  $concatfeatures  --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 0 

# CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
#                --epoch $epoch --nruns $nruns  --evalepochs 50  --outname ${model}_nognn_bs128_emb128 --embsize_vae 128 \
#                --batchsize 128  $concatfeatures  --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 0 

# CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
#                --epoch $epoch --nruns $nruns  --evalepochs 50  --outname ${model}_nognn_bs128_emb256 --embsize_vae 256 \
#                --batchsize 128  $concatfeatures  --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 0 --skip_preclustering




CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
              $common_params --epoch $epoch --nruns $nruns  --evalepochs 50  --outname ${model}_nognn_bs128 \
              --batchsize 128  $concatfeatures  --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 0 

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 50  --outname ${model}_nognn_bs256 \
               --batchsize 256  $concatfeatures  --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 0

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 50  --outname ${model}_nognn_bs512 \
               --batchsize 512  $concatfeatures  --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 0 

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 50  --outname ${model}_nognn_bs1024 \
               --batchsize 1024  $concatfeatures  --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 0 

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 50  --outname ${model}_nognn_bs2048 \
               --batchsize 2048  $concatfeatures  --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 0 

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name $model \
              $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname $model \
              $common_params --batchsize 256 $concatfeatures --rawfeatures --eval_split $evalsplit --evalskip $evalskip   

# lr search
CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name $model \
            $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_lre-1 \
             $concatfeatures --lr_gnn 1e-1 --rawfeatures --eval_split $evalsplit --evalskip $evalskip 

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
              $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_lre-2 \
                $concatfeatures --lr_gnn 1e-2  --rawfeatures --eval_split $evalsplit --evalskip $evalskip

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_lre-3 \
                $concatfeatures --lr_gnn 1e-3 --rawfeatures --eval_split $evalsplit --evalskip $evalskip  

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_lre-4 \
                $concatfeatures --lr_gnn 1e-4 --rawfeatures --eval_split $evalsplit --evalskip $evalskip  

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               --markers marker_gene_stats.tsv --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_lre-5 \
                $concatfeatures --lr_gnn 1e-5 --rawfeatures --eval_split $evalsplit --evalskip $evalskip  

# SGC loss weight
CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_scg0.1 \
                $concatfeatures --rawfeatures --eval_split $evalsplit --evalskip $evalskip --scg_alpha 0.1  

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_scg1 \
                $concatfeatures --rawfeatures --eval_split $evalsplit --evalskip $evalskip --scg_alpha 1  

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_scg10 \
                $concatfeatures --rawfeatures --eval_split $evalsplit --evalskip $evalskip --scg_alpha 10  



CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_scg1000 \
                $concatfeatures --rawfeatures --eval_split $evalsplit --evalskip $evalskip --scg_alpha 1000  

# gnn loss
CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_gnn0.01 \
                 $concatfeatures   --rawfeatures --eval_split $evalsplit --evalskip $evalskip --gnn_alpha 0.01  
CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_gnn0.1 \
                 $concatfeatures   --rawfeatures --eval_split $evalsplit --evalskip $evalskip --gnn_alpha 0.1  
CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_gnn0.3 \
                 $concatfeatures   --rawfeatures --eval_split $evalsplit --evalskip $evalskip --gnn_alpha 0.3  
CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_gnn0.5 \
               $concatfeatures  --rawfeatures --eval_split $evalsplit --evalskip $evalskip --gnn_alpha 0.5  
CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_gnn2 \
               $concatfeatures   --rawfeatures --eval_split $evalsplit --evalskip $evalskip --gnn_alpha 2  

#dropout

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_dropout0.1 \
                $concatfeatures    --rawfeatures --eval_split $evalsplit --evalskip $evalskip --dropout_gnn 0.1  

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_dropout0.5 \
                 $concatfeatures   --rawfeatures --eval_split $evalsplit --evalskip $evalskip --dropout_gnn 0.5  

# # layers_gnn
CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
              $common_params --epoch $epoch --nruns $nruns  --evalepochs 50  --outname ${model}_nognn \
               --embsize_vae 32 --embsize_gnn 32 --batchsize 256  $concatfeatures  --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 0  

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns 1  --evalepochs 20  --outname ${model}_layers1 \
               --embsize_vae 32 --embsize_gnn 32 --batchsize 256  $concatfeatures   --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 1  

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns 1  --evalepochs 20  --outname ${model}_layers2 \
               --embsize_vae 32 --embsize_gnn 32 --batchsize 256 $concatfeatures    --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 2  

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns 1  --evalepochs 20  --outname ${model}_layers4 \
               --embsize_vae 32 --embsize_gnn 32 --batchsize 256  $concatfeatures   --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 4  

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns 1  --evalepochs 20  --outname ${model}_layers5 \
              --embsize_vae 32 --embsize_gnn 32  --batchsize 256  $concatfeatures   --rawfeatures --eval_split $evalsplit --evalskip $evalskip --layers_gnn 5  

# # layer size
CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_h256 \
               --batchsize 256  $concatfeatures   --rawfeatures --eval_split $evalsplit --evalskip $evalskip --hidden_gnn 256  

CUDA_VISIBLE_DEVICES=$CUDA python src/graphmb/main.py --cuda --assembly $DATADIR --outdir $RESULTSDIR --model_name ${model} \
               $common_params --epoch $epoch --nruns $nruns  --evalepochs 20  --outname ${model}_h512 \
               --batchsize 256  $concatfeatures   --rawfeatures --eval_split $evalsplit --evalskip $evalskip --hidden_gnn 512    