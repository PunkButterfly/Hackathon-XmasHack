import re
import streamlit as st
from annotated_text import annotated_text


def process_input(text):
    result = re.sub(r"\<[\s\S]*?\>", "", text)
    result = re.sub(r"\n*\n", "\n", result)
    result = re.sub(r"\_*\_", "___", result)
    result = re.sub(r"\n\_*", "\n", result)

    return result


def view_document(document_text, ids, color="#1C6758"):
    processed_text = process_input(document_text)
    text_splitted = re.split(r"(\d+[0-9\.]*\.\s)", processed_text)

    new_splitting = [text_splitted[0]]
    for index in range(1, len(text_splitted[1:-1]), 2):
        new_splitting.append(text_splitted[index] + text_splitted[index + 1])

    best_factors_ids = ids
    colored_factors = []
    for index in range(len(new_splitting)):
        if index in best_factors_ids:
            colored_factors.append((new_splitting[index], "", color))  #  "#1C6758" "#25316D" "#4C0033" "#4C3A51" "#630606"
        else:
            colored_factors.append(new_splitting[index])

    st.subheader("Просмотр документа")

    for index in range(len(colored_factors)):
        annotated_text(colored_factors[index])