from pathlib import Path
import gdown
import yaml
import torch
from ml_model.model import BertForSequenceClassification
from transformers import AutoTokenizer
import streamlit as st


@st.cache
def download_model():
    file_name = "ml_model/DeepPavlov-Ru-bert-fine-tuned.pt"
    f_checkpoint = Path(file_name)
    url = 'https://drive.google.com/file/d/16HXz-Ust9GUrssAHLQj27GTZowONXBuw/view?usp=share_link'

    if not f_checkpoint.exists():
        gdown.download(url, file_name, fuzzy=True)

    with open("./config.yml", "r") as yamlfile:
        cfg = yaml.safe_load(yamlfile)
        print("Read successful")

    tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

    model = BertForSequenceClassification(
        pretrained_model_name='DeepPavlov/rubert-base-cased',
        num_labels=cfg['model']['num_classes'],  # 5
        dropout=cfg['model']['dropout'],  # 0.25
    )
    model.load_state_dict(torch.load("ml_model/DeepPavlov-Ru-bert-fine-tuned.pt", map_location=torch.device('cpu')))

    return model
