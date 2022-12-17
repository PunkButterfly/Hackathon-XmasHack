import streamlit as st
import numpy as np
import re

from inference import *
from view_document import view_document
from process_files import convert_file_to_text
from annotated_text import annotated_text

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

document_content = None
output = None

if 'output' not in st.session_state:
    st.session_state.output = output

analyze_button = st.button("Анализ", key="predict")

if "analyze_button_state" not in st.session_state:
    st.session_state.analyze_button_state = False

if analyze_button:

    document_content = None

    if document_text:
        document_content = document_text
    elif document_file:
        document_content = convert_file_to_text(document_file)
    st.session_state.document_content = document_content

    print(f"текст документы: {document_content}")
    output = run_inference(document_content, device='cpu')
    st.session_state.output = output

    # st.write(f"Most confident label: {output[0]}")
    # st.write(f"Type: {reversed_mapping[output[0]]}")
    # st.write(f"Factors: {sorted(output[2])}")
    #
    # st.subheader("Количество голосов")
    # for key, value in output[1][1].items():
    #     st.write(f"{key}: {value}")

viewing_column, controlling_column = st.columns(2, gap="large")

output = st.session_state.output
with controlling_column:
    colors = ["#1C6758", "#25316D", "#4C0033", "#4C3A51", "#630606"]

    st.write("")
    class_id = 0
    annotated_text(
        (
            f"{reversed_mapping[list(output[3].items())[class_id][0]]} - {int(list(output[3].items())[class_id][1] * 100)}%",
            "", colors[class_id])
    )
    view_first_class = st.button("Выделить", key=f"view_first")

    st.write("")
    class_id = 1
    annotated_text(
        (
            f"{reversed_mapping[list(output[3].items())[class_id][0]]} - {int(list(output[3].items())[class_id][1] * 100)}%",
            "", colors[class_id])
    )
    view_second_class = st.button("Выделить", key=f"view_second")

    st.write("")
    class_id = 2
    annotated_text(
        (
            f"{reversed_mapping[list(output[3].items())[class_id][0]]} - {int(list(output[3].items())[class_id][1] * 100)}%",
            "", colors[class_id])
    )
    view_third_class = st.button("Выделить", key=f"view_third")

    st.write("")
    class_id = 3
    annotated_text(
        (
            f"{reversed_mapping[list(output[3].items())[class_id][0]]} - {int(list(output[3].items())[class_id][1] * 100)}%",
            "", colors[class_id])
    )
    view_fourth_class = st.button("Выделить", key=f"view_fourth")

    st.write("")
    class_id = 4
    annotated_text(
        (
            f"{reversed_mapping[list(output[3].items())[class_id][0]]} - {int(list(output[3].items())[class_id][1] * 100)}%",
            "", colors[class_id])
    )
    view_fifth_class = st.button("Выделить", key=f"view_fifth")

with viewing_column:
    if view_first_class:
        view_document(st.session_state.document_content, output[2].get(0, np.array([])).tolist(), colors[0])
    elif view_second_class:
        view_document(st.session_state.document_content, output[2].get(1, np.array([])).tolist(), colors[1])
    elif view_third_class:
        view_document(st.session_state.document_content, output[2].get(2, np.array([])).tolist(), colors[2])
    elif view_fourth_class:
        view_document(st.session_state.document_content, output[2].get(3, np.array([])).tolist(), colors[3])
    elif view_fifth_class:
        view_document(st.session_state.document_content, output[2].get(4, np.array([])).tolist(), colors[4])

'''
streamlit run main.py
'''
