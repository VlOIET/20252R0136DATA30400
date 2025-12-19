#!/bin/bash
set -e

cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)

# python src/01_data_preprocessing.py
# python src/02_bert_embedding.py
python src/03_GAT_class.py