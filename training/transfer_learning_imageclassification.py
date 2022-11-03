import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

%matplotlib inline
import os
os.getcwd()
# place the files in your IDE working dicrectory .
labels = pd.read_csv(r'/aerialcactus/train.csv')
submission = pd.read_csv(r'/aerialcactus/sample_submission.csv)

train_path = r'/aerialcactus/train/train/'
test_path = r'/aerialcactus/test/test/'
from transformers import SegformerFeatureExtractor, SegformerForImageClassification
from PIL import Image
import requests

                         
labels.head()
labels.tail()                         
labels['has_cactus'].value_counts()
label = 'Has Cactus', 'Hasn\'t Cactus'
plt.figure(figsize = (8,8))
plt.pie(labels.groupby('has_cactus').size(), labels = label, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()
                         
root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
root.addHandler(handler)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING")) #dataset
    # parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    # parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    parser.add_argument("--pretrained-model", type=str, default=os.environ.get("SM_CHANNEL_MODEL"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--adam-learning-rate", type=float, default=5e-5)
    return parser.parse_known_args()

def _obtain_data(data_dir: str):
    """Return pytorch data train and test tuple.

    Args:
        data_dir: directory where the .py data file is loaded.

    Returns:
        Tuple: pytorch data objects
    """

    tokenized_train_dataset, tokenized_eval_dataset = _prepare_data(data_dir)

    return tokenized_train_dataset, tokenized_eval_dataset

def _get_model_and_tokenizer(args) -> Tuple[AutoModelForQuestionAnswering, AutoTokenizer]:
    """Extract model files and load model and tokenizer.

    Args:
        args: parser argument
    Returns:
        object: a tuple of tokenizer and model.
    """
    pretrained_model_path = next(pathlib.Path(args.pretrained_model).glob(constants.TAR_GZ_PATTERN))

    # extract model files
    with tarfile.open(pretrained_model_path) as saved_model_tar:
        saved_model_tar.extractall(".")
    # load model and tokenizer
    model_checkpoint = constants.MODEL_NAME_DIR
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    return model, tokenizer


def _compute_metrics(pred) -> dict:
    """Computes accuracy, precision, and recall.
    This function is a callback function that the Trainer calls when evaluating.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    print("=================f1====================")
    print(f1)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def run_with_args(args):
    """Run training."""
    model, tokenizer = _get_model_and_tokenizer(args=args)

    # train_dataset, eval_dataset = _prepare_data(args.train, tokenizer)
    train_dataset, eval_dataset = _obtain_data(args.train) # args.train 指数据集目录


    logging.info(f" loaded train_dataset sizes is: {len(train_dataset)}")
    logging.info(f" loaded eval_dataset sizes is: {len(eval_dataset)}")

    # define training args
    training_args = TrainingArguments(
        output_dir=".",
        save_total_limit=1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy = "epoch",
        logging_dir=".",
        learning_rate=float(args.adam_learning_rate),
        # load_best_model_at_end=False, # Load the optimal model after training is complete
        #metric_for_best_model="f1", #The metric to use to compare two different models."
        disable_tqdm=True,
        logging_first_step=True,
        logging_steps=50,
    )
    
    # create Trainer instance (其实一边training 一边比较eval)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=_compute_metrics,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
    )

    # it executes the training loop and saves the best model.
    trainer.train()

    # Saves the model to s3
    trainer.model.save_pretrained(args.model_dir)
    trainer.tokenizer.save_pretrained(args.model_dir)
    with open(os.path.join(args.model_dir, constants.LABELS_INFO), "w") as nf:
        nf.write(json.dumps({constants.LABELS: [0, 1]}))

if __name__ == "__main__":
    args, unknown = _parse_args()
    run_with_args(args)

