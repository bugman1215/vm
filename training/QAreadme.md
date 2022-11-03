# Lab5: fine-tune model automatically

## Training distilbert-base-cased model automatically

### Pull docker image as running environment
```
docker pull 10.117.7.210/dlc/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-jupyter-nlp
```

### Download distilbert-base-cased model example 
```
git clone https://github.com/AmyHoney/ml-models-group-learning.git
cd ml-models-group-learning/lab5/model-targz
git clone  https://gitlab.eng.vmware.com/models/pytorch-eqa-distilbert-base-cased.git
tar -czvf pytorch-eqa-distilbert-base-cased.tar.gz pytorch-eqa-distilbert-base-cased

```

### Mount lab5 exprimental and run docker container
```
sudo docker run -it --gpus all -v ~/ml-models-group-learning:/ml-models-group-learning 10.117.7.210/dlc/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-jupyter-nlp
```

### Start to train model
```
docker exec -it <contianer_id> bash
cd ml-models-group-learning/lab5
python transfer_learning_qa.py \
    --model-dir "./distilbert_model_dir" \
    --train "/ml-models-group-learning/lab5/datasets/squad/" \
    --pretrained-model "/ml-models-group-learning/lab5/model-targz"  \
    --epochs 1  \
    --batch-size 4
```

## Training other QuestionAnswer model automatically

### Prepare model

```
git-lfs clone https://huggingface.co/bert-base-cased

# Please notice: delete .git and .gitattributes files. Or get error when you train model：if this is a private repository, make sure to pass a token having permission to this repo with `use_auth_token` or log in with `huggingface-cli login` and pass `use_auth_token=True`.

cd bert-base-cased
rm -rf .git
rm -rf .gitattributes

# Archiver to tar.gz
tar -czvf bert-base-cased.tar.gz bert-base-cased
```

### Prepare dataset

```
# https://huggingface.co/datasets/squad/tree/main
git-lfs clone https://huggingface.co/datasets/squad
```

### Update constant.py parameter

```
MODEL_NAME_DIR = "bert-base-cased" #Change to your model's name
INPUT_DATA_FILENAME_PY = "squad.py" # Change to your dataset's script
```

### start to training

```
python transfer_learning_qa.py 
    --model-dir "./bert_model_dir" 
    --train "/pytorch/fine-tunable/01-fine-tunable/script/sourcedir/datasets/squad/" 
    --pretrained-model "/pytorch/fine-tunable/model-targz" --epochs 1 
    --batch-size 16
```
