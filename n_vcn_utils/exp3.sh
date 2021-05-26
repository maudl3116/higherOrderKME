#!/bin/bash
for r in $(seq 1 1 15)
do
  for i in $(seq 10 10 50)
  do
    python dataset.py \
        --env Ball \
        --stage dy \
        --n_rollout 100 \
        --n_ball 5 \
        --time_step $i \
        --node_attr_dim 0 \
        --edge_attr_dim 1 \
        --edge_type_num 3 \
        --edge_st_idx 1 \
        --edge_share 1 \
        --gen_data 1 \
        --num_workers 1 \
        --rel_type 1 \
        --rel_attr 30 \
        --h5 1 \
        --dataf "exp3/$r"
  done
done




