import os
from constants import constants
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_train_features(examples):

    # remove the left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride.
    # tokenizer = AutoTokenizer.from_pretrained("./pytorch-eqa-distilbert-base-cased")
    model_checkpoint = constants.MODEL_NAME_DIR
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    pad_on_right = tokenizer.padding_side == "right"

    tokenized_train_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=constants.MAX_SEQ_LENGTH,
        stride=constants.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_train_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_train_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_train_examples["start_positions"] = []
    tokenized_train_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_train_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_train_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_train_examples["start_positions"].append(cls_index)
            tokenized_train_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_train_examples["start_positions"].append(cls_index)
                tokenized_train_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_train_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_train_examples["end_positions"].append(token_end_index + 1)

    return tokenized_train_examples

def prepare_validation_features(examples):
    # remove the left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. 
    # tokenizer = AutoTokenizer.from_pretrained("./pytorch-eqa-distilbert-base-cased")
    model_checkpoint = constants.MODEL_NAME_DIR
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    pad_on_right = tokenizer.padding_side == "right"

    tokenized_val_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=constants.MAX_SEQ_LENGTH,
        stride=constants.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_val_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_val_examples["example_id"] = []

    for i in range(len(tokenized_val_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_val_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_val_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_val_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_val_examples["offset_mapping"][i])
        ]

    return tokenized_val_examples
 
def _prepare_data(data_dir: str):
    """Return pytorch data train and test tuple.
    Args:
        data_dir: directory where the .py data file is loaded.
        tokenizer: tokenizer from the huggingface library.
    Returns:
        Tuple: pytorch data objects
    """

    # load dataset from py file
    dataset = load_dataset(
        os.path.join(data_dir, constants.INPUT_DATA_FILENAME_PY),
        constants.INPUT_DATA_CINFIG
    )

    # preprocess dataset
    tokenized_dataset = dataset.map(
        prepare_train_features, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )

    tokenized_eval_dataset = dataset["validation"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=dataset["validation"].column_names
    )

    tokenized_train_dataset = tokenized_dataset["train"]
    tokenized_eval_dataset = tokenized_eval_dataset

    # train_test_split dataset
    print("=================train====================")
    print(tokenized_train_dataset)
    print("=================validation====================")
    print(tokenized_eval_dataset)

    # return preprocessed_dataset[constants.TRAIN], preprocessed_dataset[constants.VALIDATION]
    return tokenized_train_dataset, tokenized_eval_dataset

def _prepare_data_from_json(data_dir: str):
    """Return pytorch data train and test tuple.
    Args:
        data_dir: directory where the .py data file is loaded.
        tokenizer: tokenizer from the huggingface library.
    Returns:
        Tuple: pytorch data objects
    """

    # load dataset from py file
    dataset = load_dataset(
        os.path.join(data_dir, constants.INPUT_DATA_FILENAME_PY),
        constants.INPUT_DATA_CINFIG
    )

    # preprocess dataset
    tokenized_dataset = dataset.map(
        prepare_train_features, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )

    tokenized_eval_dataset = dataset["validation"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=dataset["validation"].column_names
    )

    tokenized_train_dataset = tokenized_dataset["train"]
    tokenized_eval_dataset = tokenized_eval_dataset

    # train_test_split dataset
    print("=================train====================")
    print(tokenized_train_dataset)
    print("=================validation====================")
    print(tokenized_eval_dataset)

    # return preprocessed_dataset[constants.TRAIN], preprocessed_dataset[constants.VALIDATION]
    return tokenized_train_dataset, tokenized_eval_dataset
Footer
© 2022 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
