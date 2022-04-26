import sys
import json
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union

import datasets
import numpy as np
import torch
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

if __name__ == "__main__":
    model_path = sys.argv[1]
    test_file = sys.argv[2]
    swag_test_file = 'cs_test.csv'

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMultipleChoice.from_pretrained(model_path)

    extension = 'csv'
    raw_dataset = load_dataset(
        extension,
        data_files=swag_test_file,
        cache_dir=None,
        use_auth_token=False
    )

 # When using your own dataset or a different dataset from swag, you will probably need to change this.
    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"
    
    max_seq_length = tokenizer.model_max_length

    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        # Un-flatten
        return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    test_datasets = raw_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=None,
    )
    
    pad_to_max_length = False
    # Data collator
    data_collator = (
        default_data_collator
        if pad_to_max_length
        else DataCollatorForMultipleChoice(tokenizer=tokenizer, pad_to_multiple_of=None)
    )

    trainer = Trainer(
        model=model,
        train_dataset = None,
        eval_dataset = None,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    predictions = trainer.predict(test_dataset=test_datasets['train']).predictions
    predictions = np.argmax(predictions, axis=1)
    
    f = open(test_file,encoding='utf-8')
    test_datas = json.load(f)
    f.close()

    for i, d in enumerate(test_datas):
        d['relevant'] = d['paragraphs'][predictions[i]]

    with open(f'qa_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_datas, f, ensure_ascii=False, indent=4)