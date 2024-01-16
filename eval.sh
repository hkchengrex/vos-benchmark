#!/bin/bash

path_list=(
    "/data1/zhangjiaming/results_Tracking_Anything/v4_convmae_mask2former/1230-1-fix.yaml/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6_200000.pth/TrackingAnything_1230-test.txt-memory_store_every{5}-memory_max_number{1}-model_topk{30}"
    "/data1/zhangjiaming/results_Tracking_Anything/v4_convmae_mask2former/1230-1-fix.yaml/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6_201000.pth/TrackingAnything_1230-test.txt-memory_store_every{5}-memory_max_number{1}-model_topk{30}"
    "/data1/zhangjiaming/results_Tracking_Anything/v4_convmae_mask2former/1230-1-fix.yaml/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6_202000.pth/TrackingAnything_1230-test.txt-memory_store_every{5}-memory_max_number{1}-model_topk{30}"
    "/data1/zhangjiaming/results_Tracking_Anything/v4_convmae_mask2former/1230-1-fix.yaml/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6_203000.pth/TrackingAnything_1230-test.txt-memory_store_every{5}-memory_max_number{1}-model_topk{30}"
    "/data1/zhangjiaming/results_Tracking_Anything/v4_convmae_mask2former/1230-1-fix.yaml/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6_204000.pth/TrackingAnything_1230-test.txt-memory_store_every{5}-memory_max_number{1}-model_topk{30}"
    "/data1/zhangjiaming/results_Tracking_Anything/v4_convmae_mask2former/1230-1-fix.yaml/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6_205000.pth/TrackingAnything_1230-test.txt-memory_store_every{5}-memory_max_number{1}-model_topk{30}"
    "/data1/zhangjiaming/results_Tracking_Anything/v4_convmae_mask2former/1230-1-fix.yaml/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6_206000.pth/TrackingAnything_1230-test.txt-memory_store_every{5}-memory_max_number{1}-model_topk{30}"
    "/data1/zhangjiaming/results_Tracking_Anything/v4_convmae_mask2former/1230-1-fix.yaml/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6_207000.pth/TrackingAnything_1230-test.txt-memory_store_every{5}-memory_max_number{1}-model_topk{30}"
    "/data1/zhangjiaming/results_Tracking_Anything/v4_convmae_mask2former/1230-1-fix.yaml/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6_208000.pth/TrackingAnything_1230-test.txt-memory_store_every{5}-memory_max_number{1}-model_topk{30}"
    "/data1/zhangjiaming/results_Tracking_Anything/v4_convmae_mask2former/1230-1-fix.yaml/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6_209000.pth/TrackingAnything_1230-test.txt-memory_store_every{5}-memory_max_number{1}-model_topk{30}"
    "/data1/zhangjiaming/results_Tracking_Anything/v4_convmae_mask2former/1230-1-fix.yaml/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6/Jan08_06.08.25_v4_convmae_mask2former_1230-1-fix_s6_210000.pth/TrackingAnything_1230-test.txt-memory_store_every{5}-memory_max_number{1}-model_topk{30}"
)

for path in "${path_list[@]}"; do
    echo "Current path: $path"

    python benchmark.py \
    -n 16 \
    -g /data1/zhangjiaming/datasets/Tracking_Anything/Tracking_Anything_Dataset/first_stage/Annotations \
    -m $path \
    -v /data1/zhangjiaming/datasets/Tracking_Anything/Tracking_Anything_Dataset/first_stage/ImageSets/1230-test.txt
done