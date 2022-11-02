24 lines (19 sloc)  710 Bytes

MODEL_NAME_DIR = "resnet-50" 
# MODEL_NAME_DIR = "resnet-tiny-beans"
TRAIN_VAL_SPLIT = 0.80
# INPUT_DATA_FILENAME = "data.csv"
INPUT_DATA_FILENAME_PY = "squad.py" # squad.py, adversarial_qa.py
INPUT_DATA_CINFIG = "plain_text" # # if adversarial_qa dataset, need to give valid config: `adversarialQA`; if squad dataset, need to give valid config: `plain_text`.

INPUT_DATA_FILENAME_JSON = "dureader_robust.train.json"

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
TAR_GZ_PATTERN = "*.tar.gz"

MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128

SENTENCES1 = "sentences1"
SENTENCES2 = "sentences2"
QUESTION = "question"
CONTEXT = "context"
LABELS = "labels"

LABELS_INFO = "labels_info.json"
