#!/bin/bash

source /home/amueller/miniconda3/bin/activate
conda activate xtreme

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`

# python attention_intervention_structural.py --gpt2-version gpt2-medium --do-filter True --structure simple_agreement
python attention_intervention_structural.py --gpt2-version gpt2-medium --do-filter True --structure across_obj_rel --filter-quantile 0.8
# python attention_intervention_structural.py --gpt2-version gpt2-medium --do-filter True --structure across_subj_rel --filter-quantile 0.8
# python attention_intervention_structural.py --gpt2-version gpt2-medium --do-filter True --structure within_obj_rel --filter-quantile 0.5
