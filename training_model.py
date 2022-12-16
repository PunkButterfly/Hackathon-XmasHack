from typing import List, Mapping, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AutoTokenizer

torch.manual_seed(42)


class TextClassificationDataset(Dataset):

    def __init__(
            self,
            texts: List[str],
            labels: List[int] = None,
            labels_dict: Mapping[str, str] = None,
            max_seq_length: int = 128,
            model_name: str = None,
            add_spec_tokens: bool = False,
    ):
        """
        Args:
            texts: a list of text to classify
            labels: a list with classification labels
            labels_dict: a dictionary mapping class names to class ids
            max_seq_length: maximum sequence length in tokens
            model_name: transformer model name
            add_spec_tokens: if we want to add special tokens to the tokenizer
        """

        self.texts = texts
        self.labels = labels
        self.labels_dict = labels_dict
        self.max_seq_length = max_seq_length

        # labels_dict = {'class1' : '0', 'class2' : '1', 'class3' : '2', ...}
        if (self.labels_dict is None and labels is not None):
            self.labels_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # May add special tokens option later ( ['URL'], ['HASHTAG'], ['MENTION'], ['NUM'], ... )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        """
        Gets encoded representation of a single element (text) in the dataset
        Args:
            index (int): index of the element in dataset
        Returns:
            Single element by index
        """
        text = self.texts[index]

        # A dictionary with 'input_ids', 'token_type_ids', 'attention_masks' as keys
        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # encoded_dict["input_ids"]
        # encoded_dict["attention_mask"]
        # encoded_dict["token_type_ids"]
        encoded_dict["features"] = encoded_dict["input_ids"].squeeze(0)
        del encoded_dict["input_ids"]
        del encoded_dict["token_type_ids"]  # don't need them because they are for sequence-pair classification task

        # Encoding target label using our labels_dict
        if self.labels is not None:
            y = self.labels[index]
            y_encoded = torch.Tensor([self.labels_dict.get(y, -1)]).long().squeeze(0)
            encoded_dict["targets"] = y_encoded

        return encoded_dict


def read_data(cfg: dict) -> Tuple[dict, dict, dict]:
    """
    Function that reads TSV data, creates TextClassificationDatasets and DataLoaders.
    Args:
        A config dictionary containing "file_path", "max_seq_length", etc...
    Returns:
        A tuple with (training & validation), testing, unlabeled_inference DataLoader dictionaries
    """

    # Creating PyTorch Datasets
    train_dataset = TextClassificationDataset(
        texts=train_sentences['text'].values.tolist(),
        labels=train_sentences['label'].values,
        max_seq_length=cfg['model']['max_seq_length'],
        model_name=cfg['model']['model_name'],
    )

    val_dataset = TextClassificationDataset(
        texts=test_sentences['text'].values.tolist(),
        labels=test_sentences['label'].values,
        max_seq_length=cfg['model']['max_seq_length'],
        model_name=cfg['model']['model_name'],
    )

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.

    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=cfg['training']['batch_size'],
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=cfg['training']['batch_size'],
    )

    return train_loader, val_loader