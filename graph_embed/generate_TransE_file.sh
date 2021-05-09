dataset=$1
python3 convert_to_triplets.py --dataset ${dataset}

rm -rf ckpts

bash train_graph_emb.sh ${dataset}

python3 split_node_emb.py --dataset ${dataset}
mkdir -p ../TransE_${dataset}
mv *.pt ../TransE_${dataset}
