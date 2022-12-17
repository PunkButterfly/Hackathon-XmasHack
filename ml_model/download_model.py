from pathlib import Path
import gdown


def download_model():
    file_name = "ml_model/DeepPavlov-Ru-bert-fine-tuned.pt"
    f_checkpoint = Path(file_name)
    url = 'https://drive.google.com/file/d/16HXz-Ust9GUrssAHLQj27GTZowONXBuw/view?usp=share_link'

    if not f_checkpoint.exists():
        gdown.download(url, file_name, fuzzy=True)

