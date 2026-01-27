#!/bin/bash
export CUDA_VISIBLE_DEVICES="6"

run_task() {
  local dataset_name=$1
  local config_file=$2
  local work_dir="/data/ljh/segmentation/RF-CLIP/experiment/${dataset_name}"
  local show_dir="/data/ljh/segmentation/RF-CLIP/experiment/${dataset_name}/show_dir"
  local output_dir="/data2/ljh/methods/RF-CLIP/${dataset_name}/"
  
  python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=12345 \
    eval.py \
    --config "$config_file" \
    --work-dir "$work_dir" \
    --pamr off \
    --output-dir "$output_dir" \
    --launcher pytorch \
  
  sleep 2
}

run_in_batches() {
  local config_list=("$@")
  for config_file in "${config_list[@]}"; do
    local dataset_name=$(basename "$config_file" .py)
    run_task "$dataset_name" "$config_file"
  done
}

analyze_logs() {
  local config_list=("$@")
  for config_file in "${config_list[@]}"; do
    local dataset_name=$(basename "$config_file" .py)
    local log_dir="/data/ljh/segmentation/RF-CLIP/experiment/${dataset_name}"
    
    find "$log_dir" -name "*.log" -exec grep -E "data_root =|dataset_type =|mIoU: [0-9.]+" {} \;
    echo "----------------------------------------"
  done
}

configs_list=(

    # './configs/cfg_voc20.py'
    './configs/cfg_voc21.py'  
    # './configs/cfg_ade20k.py'
    # './configs/cfg_city_scapes.py'
    # './configs/cfg_coco_stuff164k.py'
    # './configs/cfg_context59.py'
    # './configs/cfg_context60.py'
    # './configs/cfg_coco_object.py'
)

main() {
  echo "===== 开始批量运行任务 ====="
  run_in_batches "${configs_list[@]}"
  
  echo "===== 开始分析日志 ====="
  analyze_logs "${configs_list[@]}"
}

main