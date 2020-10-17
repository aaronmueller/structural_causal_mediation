#!/bin/bash

source /home/amueller/miniconda3/bin/activate
conda activate xtreme

export LD_LIBRARY_PATH=/opt/NVIDIA/cuda-10/lib64
export CUDA_VISIBLE_DEVICES=`free-gpu`

python run_profession_neuron_experiments.py --grammar_file structural/grammar.avg -model gpt2 --structure simple_agreement -out_dir simple_agreement --randomize 
