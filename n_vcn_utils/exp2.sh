#!/bin/bash
for r in $(seq 1 1 15)
do
  for i in $(seq 5 1 10)
  do
    python dataset.py \
        --env Ball \
        --stage dy \
        --n_rollout 100 \
        --n_ball i \
        --time_step 20 \
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
        --dataf "exp1/$r"
  done
done




