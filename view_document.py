import re
import numpy as np
import streamlit as st
from annotated_text import annotated_text
from ml_model.inference import splitting_text_by_regex, process_text, process_splitted_text


def process_input(text):
    result = re.sub(r"\<[\s\S]*?\>", "", text)
    result = re.sub(r"\n[\s]*\n", "\n", result)
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"\_*\_", "___", result)
    result = re.sub(r"\n\_*", "\n", result)

    return result


def view_factors(document_text, ids, color="#630606"):
    splitted_text, splitting_points = splitting_text_by_regex(process_text(document_text, lower_flag=False),
                                                              splitter='([\t\n]\s*\d+[0-9\.]*\.\s)')
    splitted_text = process_splitted_text(splitted_text)
    glued_sentences = np.array(glue_sentences(splitted_text))[ids]
    colored_factors = [(item, "", color) for item in glued_sentences]

    for index in range(len(colored_factors)):
        annotated_text(colored_factors[index])


def view_document(document_text, ids, color="#630606"):
    splitted_text, splitting_points = splitting_text_by_regex(process_text(document_text, lower_flag=False),
                                                              splitter='([\t\n]\s*\d+[0-9\.]*\.\s)')
    splitted_text = process_splitted_text(splitted_text)
    # text_splitted = re.split(r"([\t\n]\s*\d+[0-9\.]*\.\s)", document_text)  # (\d+[0-9\.]*\.\s)

    glued_sentences = glue_sentences(splitted_text)

    colored_factors = []
    for index in range(len(glued_sentences)):
        if index in ids:
            colored_factors.append(
                (glued_sentences[index], "", color))  # "#1C6758" "#25316D" "#4C0033" "#4C3A51" "#630606"
        else:
            colored_factors.append(glued_sentences[index])

    for index in range(len(colored_factors)):
        annotated_text(colored_factors[index])


def glue_sentences(splitted_text):
    with_points_splitted = [splitted_text[0]]
    index = 1
    while index < len(splitted_text) - 1:
        if re.match(r"\d+[0-9\.]*\.", splitted_text[index]):
            with_points_splitted.append(splitted_text[index] + splitted_text[index + 1])
            index += 2
            continue
        else:
            with_points_splitted.append(splitted_text[index])
            index += 1

    return with_points_splitted
