#! /bin/zsh
export PATH=/cal/softs/anaconda/anaconda3/bin:$PATH
cd $HOME/prim/leaf_counting
source activate prim
python3 predict.py