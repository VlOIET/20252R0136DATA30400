#!/bin/bash
set -e

cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)

# python src/01_data_preprocessing.py
# python src/02_bert_embedding.py
# python src/03_GAT_class.py
# python src/04_make_silverlabel.py
# python src/05_train_classifier.py
python src/06_test_classifier.py