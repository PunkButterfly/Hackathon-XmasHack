import streamlit as st
from inference import *
import numpy as np
import re

from process_files import process_input, convert_file_to_text


reversed_mapping = {
    0: "Договоры оказания услуг",
    1: "Договоры купли-продажи",
    2: "Договоры аренды",
    3: "Договоры подряда",
    4: "Договоры поставки"
}
st.set_page_config(layout="wide")

st.title("Классификатор документов онлайн")

st.subheader(" ")
st.subheader("Вставьте текст документа")
document_text = st.text_area("Да-да, вот сюда :)", height=300)

st.subheader(" ")
st.subheader("Или загрузите документ в формате DOC, DOCX, PDF")
document_file = st.file_uploader("Выберите файл", type=["doc", "docx", "pdf", "rtf"])


if st.button("Узнать тип"):

    document_content = None

    if document_text:
        document_content = document_text
    elif document_file:
        document_content = convert_file_to_text(document_file)

    st.write(document_content)
    # output = run_inference(document_text, device='cpu')
    # st.write(f"Most confident label: {output[0]}")
    # st.write(f"Type: {reversed_mapping[output[0]]}")
    # # st.write(f"Factors: {sorted(output[2])}")
    #
    # st.subheader("Количество голосов")
    # for key, value in output[1][1].items():
    #     st.write(f"{key}: {value}")
    #
    # processed_text = process_input(document_text)
    # text_splitted = re.split(r"(\d+[0-9\.]*\.\s)", processed_text)
    #
    # new_splitting = [text_splitted[0]]
    # for index in range(1, len(text_splitted[1:-1]), 2):
    #     new_splitting.append(text_splitted[index] + text_splitted[index + 1])
    #
    # best_factors = np.array(new_splitting)[[sorted(output[2].tolist())]]
    #
    # st.subheader("Ключевые факторы в предсказании")
    # for factor in best_factors[0]:
    #     st.write(factor)










'''
streamlit run main.py
'''