python src/graphmb/main.py --cuda --assembly ../data/simHC/ --outdir results/simHC_edges/ --assembly_name assembly.fasta --depth
 abundance.tsv.edges_jgi  --evalskip 0 --epoch 100  --model gcn_ae --rawfeatures --batchsize 256 --gnn_alpha 0 --ae_alpha 1 --scg_alpha 0 --lr_gnn 1e-3 --layers_gnn 0 --nega
tives 10  --outname ae_lr1e-3 --nruns 1  --embsize_gnn 32