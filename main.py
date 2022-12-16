import streamlit as st
import numpy as np
import re

from inference import *
from view_document import view_document
from process_files import convert_file_to_text


reversed_mapping = {
    0: "Договоры оказания услуг",
    1: "Договоры купли-продажи",
    2: "Договоры аренды",
    3: "Договоры подряда",
    4: "Договоры поставки"
}
st.set_page_config(layout="wide")

st.title("Классификатор документов онлайн")

inputing_text_column, inputing_file_column = st.columns(2, gap="large")

with inputing_text_column:
    st.subheader(" ")
    st.subheader("Вставьте текст документа")
    document_text = st.text_area("Да-да, вот сюда :)", height=150)

with inputing_file_column:
    st.subheader(" ")
    st.subheader("Или загрузите документ DOC, DOCX, PDF, RTF")
    document_file = st.file_uploader("Выберите файл", type=["doc", "docx", "pdf", "rtf"])

with st.sidebar:
    st.write("sidebar")


if st.button("Узнать тип"):

    document_content = None

    if document_text:
        document_content = document_text
    elif document_file:
        document_content = convert_file_to_text(document_file)

    output = run_inference(document_content, device='cpu')

    st.write(f"Most confident label: {output[0]}")
    st.write(f"Type: {reversed_mapping[output[0]]}")
    st.write(f"Factors: {sorted(output[2])}")

    st.subheader("Количество голосов")
    for key, value in output[1][1].items():
        st.write(f"{key}: {value}")

    view_document(document_content, output[2].tolist()) # color











'''
streamlit run main.py
'''