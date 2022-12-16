import re
from collections import Counter
from typing import List, Mapping
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from model import BertForSequenceClassification


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


def process_text(text):
    """
    Очищает текст
    """
    text = re.sub("_+", "", text) # Удаляет подчеркивания
    text = re.sub("\([^)]*\)", "", text) # Удаляет фразы в скобках (Фамилия имя отчество)
    # text = re.sub("\([^)]*\)", "", text) # Удаляет фразы в квадратных скобках (Пока не работает)
    text = text.lower()
    # text = re.sub(r'[^\w\s]','', text)

    return text


def process_splitted_text(sequences):
    """
    Очищает список пунктов договора
    """
    result = [re.sub("\n", "", x) for x in sequences]
    result = [re.sub(r'\s+', ' ', x).strip() for x in sequences]

    return result


def splitting_text_by_regex(text, splitter="\d+[0-9\.]*\.\s"):  # \d+\.[0-9\.]* - старый вариант (резал по отсылкам на пункты) #   \n\d+[0-9\.]*
    """
    Стандартный regex сплитит по пунктам договора
    """
    # TODO: Если не находит пункты, то пусть делит по абзацам
    points = re.findall(splitter, text)
    result = re.split(splitter, text)

    return result, points


def choose(predictions, number_of_classes=5):
    """
    Посчитывает голоса для каждого класса
    """

    voices = Counter()

    for prediction in predictions:
        voices[prediction] = voices.get(prediction, 0) + 1

    return voices.most_common(number_of_classes)[0][0], voices


tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

with open("config.yml", "r") as yamlfile:
    cfg = yaml.safe_load(yamlfile)
    print("Read successful")

model = BertForSequenceClassification(
    pretrained_model_name='DeepPavlov/rubert-base-cased',
    num_labels=cfg['model']['num_classes'], # 5
    dropout=cfg['model']['dropout'], # 0.25
)
model.load_state_dict(torch.load("DeepPavlov-Ru-bert-fine-tuned.pt", map_location=torch.device('cpu')))

def run_inference(new_document_text, device, model=model):
    # Препроцесс
    splitted_text, splitting_points = splitting_text_by_regex(process_text(new_document_text))
    processed_splitted_text = process_splitted_text(splitted_text)

    splitting_points = list(range(len(splitted_text)))
    # Удаляем sequences длинны которых <= 5 (заголовки, которые одинаковые для многих документов)
    filtered_splitted_texts = []
    filtered_splitting_points = []
    # for _, text, splitter in zip(processed_splitted_text, splitting_points):
    #     if len(text.split(" ")) > 5:
    #         filtered_splitted_texts.append(text)
    #         filtered_splitting_points.append(splitter)
    for text, splitter in zip(processed_splitted_text, splitting_points):
        if len(text.split(" ")) > 5:
            filtered_splitted_texts.append(text)
            filtered_splitting_points.append(splitter)

    print(len(filtered_splitted_texts))
    print(len(filtered_splitting_points))

    # Образуем unlabeled_dataloader
    unlabeled_dataset = TextClassificationDataset(
        texts=filtered_splitted_texts,
        labels=None,
        max_seq_length=cfg['model']['max_seq_length'],
        model_name=cfg['model']['model_name'],
    )

    unlabeled_loader = DataLoader(
        dataset=unlabeled_dataset,
        sampler=SequentialSampler(unlabeled_dataset),
        batch_size=cfg['training']['batch_size'],
    )

    # The difference between eval_loop_fn is that we don't have true_labels now
    model.eval()
    final_logits = []
    tqdm_bar = tqdm(unlabeled_loader, desc="Inference", position=0, leave=True)
    for _, batch in enumerate(tqdm_bar):
        features = batch["features"]  # (input_ids)
        attention_mask = batch["attention_mask"]

        features = features.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)

        with torch.no_grad():
            outputs = model(input_ids=features,
                            attention_mask=attention_mask,
                            return_dict=True)
        final_logits.append(outputs['logits'].detach().cpu().numpy())

    # Combine the results across all batches.
    flat_predictions = np.concatenate(final_logits, axis=0)
    predicted_labels = np.argmax(flat_predictions,
                                 axis=1).flatten()  # perform argmax for each sample to output labels, not scores

    # return most common labels Counter
    most_confident_labels = choose(predicted_labels)  # [(2, 52), (4, 9), (0, 8), (3, 5)]
    most_confident_label = most_confident_labels[0]
    print(f"labels:{most_confident_labels}")
    print(f"Most confident label is {most_confident_label}")
    # must be equal to number of sequences
    # print(len(final_logits))  # equals to number of batches
    # for i in range(len(final_logits)):
    best_sequences = flat_predictions[np.argmax(flat_predictions, axis=1).flatten() == most_confident_label]
    beam_size = 10
    best_sentences_ids = np.argpartition(best_sequences[:, 2], -beam_size)[-beam_size:][::-1]

    return most_confident_label, most_confident_labels, best_sentences_ids, filtered_splitting_points, processed_splitted_text, final_logits