from __future__ import annotations
import streamlit as st
import re
from bs4 import BeautifulSoup
from ml_model import *
# import os
# import PyPDF2
# from PyPDF2 import PdfFileReader
# import docx
import pandas as pd
import numpy as np
import json
# import aspose.words as aw
# import traceback
import re


def read_docx_file(docx_file_path: str) -> str:
    '''
        Открываем .docx файл и, соединяя все абзацы,
        возращем полный текст документа
        или None если не удалось открыть файл
    '''
    try:
        doc = docx.Document(docx_file_path)
        all_paras = doc.paragraphs
        full_doc_text = ''
        for para in all_paras:
            full_doc_text = full_doc_text + '\n' + para.text
        return full_doc_text.replace('Evaluation Only. Created with Aspose.Words. Copyright 2003-2022 Aspose Pty Ltd.', '')
    except:
        print(f'Ошибка: Не удалось открыть файл: {docx_file_path}\n {traceback.format_exc()}')
        return None


def process_input(text):
    result = re.sub(r"\<[\s\S]*?\>", "", text)
    result = re.sub(r"\n*\n", "\n", result)
    result = re.sub(r"\_*\_", "___", result)
    result = re.sub(r"\n\_*", "\n", result)

    return result

st.set_page_config(layout="wide")

st.title("Классификатор документов онлайн")
st.subheader(" ")

st.subheader("Вставьте текст документа")

document_text = st.text_area("Да-да, вот сюда :)", height=400)

st.subheader(" ")
st.subheader("Или загрузите документ в формате DOC, DOCX, PDF")

document_file = st.file_uploader("Выберите файл")
# document_text_from_file = read_docx_file()

st.write("___\n")

reversed_mapping = {
    0: "Договоры для акселератора/Договоры оказания услуг",
    1: "Договоры для акселератора/Договоры купли-продажи",
    2: "Договоры для акселератора/Договоры аренды",
    3: "Договоры для акселератора/Договоры подряда",
    4: "Договоры для акселератора/Договоры поставки"
}

if st.button("Узнать тип"):
    output = run_inference(document_text, device='cpu')
    st.write(f"Most confident label: {output[0]}")
    st.write(f"Type: {reversed_mapping[output[0]]}")
    # st.write(f"Factors: {sorted(output[2])}")

    st.subheader("Количество голосов")
    for key, value in output[1][1].items():
        st.write(f"{key}: {value}")

    processed_text = process_input(document_text)
    text_splitted = re.split(r"(\d+[0-9\.]*\.\s)", processed_text)

    new_splitting = [text_splitted[0]]
    for index in range(1, len(text_splitted[1:-1]), 2):
        new_splitting.append(text_splitted[index] + text_splitted[index + 1])

    best_factors = np.array(new_splitting)[[sorted(output[2].tolist())]]

    st.subheader("Ключевые факторы в предсказании")
    for factor in best_factors[0]:
        st.write(factor)










'''
streamlit run main.py
'''