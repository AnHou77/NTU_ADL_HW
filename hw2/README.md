# Homework 2 ADL NTU 110 Spring

## Environment & Package
```shell
# Environment
Python 3.8

# Package
PyTorch 1.10.2
transformers 4.17.0
datasets 2.0.0
scikit-learn 1.0.2
tqdm
numpy
pandas
```

## Inference
```shell
# download model
bash download.sh

# inference with Multiple Choice & Question Answering model 
# "${1}": path to the context file.
# "${2}": path to the testing file.
# "${3}": path to the output predictions.
bash run.sh ${1} ${2} ${3}
```
## Train
### Preprocess
```
# make a data structure like:
- data/
    |- train.json
    |- valid.json
```
### Multiple Choice
```
# 將 data 轉換成 swag 格式
python preprocess_context_train.py

# training
# ${output_dir} can be any path
python run_swag.py \
    --model_name_or_path bert-base-chinese \
    --train_file data/swag_train.csv \
    --validation_file data/swag_valid.csv \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 512 \
    --output_dir ${output_dir} \
    --gradient_accumulation_steps 2
```
### Question Answering
```
# 將 data 轉換成 squad 格式
python preprocess_qa_train.py

# training & predict
## ${output_dir} can be any path
python run_qa.py \
    --model_name_or_path hfl/chinese-bert-wwm-ext \
    --train_file data/squad_train.json \
    --validation_file data/squad_valid.json \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir ${output_dir}
```