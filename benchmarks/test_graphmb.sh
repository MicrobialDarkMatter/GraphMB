#!/bin/bash
DATADIR=data/

#check data
if [ -d $DATADIR/strong100 ] 
then
    echo "Strong100 dataset found" 
else
    echo "Error: dataset not found, downloading"
    cd $DATADIR; wget https://zenodo.org/record/6122610/files/strong100.zip; unzip strong100.zip
fi

# check venv
if [ -d "./venv/" ] 
then
    echo "venv found" 
    source venv/bin/activate
else
    echo "venv not found"
    python -m venv venv
    source venv/bin/activate
    pip install -e .
fi

python src/graphmb/main.py --assembly $DATADIR/strong100/ --outdir results/strong100/ --markers marker_gene_stats.tsv