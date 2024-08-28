#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
# export CUDA_VISIBLE_DEVICES= 0

./tools/test_net.py \
  --network seg_resnet34_8s_embedding \
  --pretrained /home/user/scx/Code/UnseenObjectClustering/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth  \
  --dataset ocid_object_test \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
  --pretrained_crop /home/user/scx/Code/UnseenObjectClustering/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth  \
