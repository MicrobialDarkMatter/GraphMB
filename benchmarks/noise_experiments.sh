


python src/graphmb/main.py --assembly ../data/strong100/ --outdir results/simdata/ --evalskip 200 --epoch 1000 \
                           --model gcn_ae --rawfeatures --gnn_alpha 1 --ae_alpha 0 --scg_alpha 1 --lr_gnn 1e-3 \
                           --layers_gnn 1 --read_cache --markers "" --scg_alpha 0 --noise

python src/graphmb/main.py --assembly ../data/strong100/ --outdir results/strong100/ --evalskip 200 --epoch 1000 \
                           --model gcn_ae --rawfeatures --gnn_alpha 1 --ae_alpha 0 --scg_alpha 1 --lr_gnn 1e-3 \
                           --layers_gnn 1 --scg_alpha 0 --noise