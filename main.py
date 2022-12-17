import streamlit as st
from ml_model.inference import *
from features import get_confidence_sentences_ids, get_entities
from view_document import view_document, view_factors
from process_files import convert_file_to_text
from annotated_text import annotated_text
from ml_model.download_model import download_model

colors = ["#1C6758", "#25316D", "#4C0033", "#4C3A51", "#630606"]
reversed_mapping = {
    0: "Договоры оказания услуг",
    1: "Договоры купли-продажи",
    2: "Договоры аренды",
    3: "Договоры подряда",
    4: "Договоры поставки"
}
st.set_page_config(layout="wide")

model = download_model()

st.title("Классификатор документов")

inputing_text_column, inputing_file_column = st.columns(2, gap="large")

with inputing_text_column:
    st.subheader(" ")
    st.subheader("Вставьте текст документа")
    document_text = st.text_area("Да-да, вот сюда :)", height=150)

with inputing_file_column:
    st.subheader(" ")
    st.subheader("Или загрузите документ DOC, DOCX, PDF, RTF")
    document_file = st.file_uploader("", type=["doc", "docx", "pdf", "rtf"])

analyze_button = st.button("Анализ", key="predict")

if 'output' not in st.session_state:
    st.session_state['output'] = False

if analyze_button:

    if document_text:
        document_content = document_text

        st.session_state.document_content = document_content

        output = run_inference(document_content, device='cpu', model=model)
        st.session_state.output = output
    elif document_file:
        document_content = convert_file_to_text(document_file)

        st.session_state.document_content = document_content

        output = run_inference(document_content, device='cpu', model=model)
        st.session_state.output = output

st.write("___")
st.write(" ")
viewing_column, controlling_column = st.columns(2, gap="large")

try:
    if not st.session_state.output:
        raise ValueError

    output = st.session_state.output

    with controlling_column:
        st.subheader("Ключевые факторы")

        confidence_filter = st.slider("Уровень доверия", min_value=0.1, max_value=0.9, step=0.1, value=0.8)
        st.session_state.confidence_filter = confidence_filter

        st.write(" ")
        class_id = 0
        annotated_text(
            (
                f"{reversed_mapping[list(output[3].items())[class_id][0]]} - {int(list(output[3].items())[class_id][1] * 100)}%",
                "", colors[class_id])
        )
        view_first_class = st.button("Выделить", key=f"view_first")
        if view_first_class:
            ids = get_confidence_sentences_ids(*output[2], quantile_param=confidence_filter).get(0, np.array([])).tolist()
            view_factors(st.session_state.document_content,
                         ids[0:min(len(ids), 15)], colors[0])

        st.write("___")
        st.write("")
        class_id = 1
        annotated_text(
            (
                f"{reversed_mapping[list(output[3].items())[class_id][0]]} - {int(list(output[3].items())[class_id][1] * 100)}%",
                "", colors[class_id])
        )
        view_second_class = st.button("Выделить", key=f"view_second")
        if view_second_class:
            ids = get_confidence_sentences_ids(*output[2], quantile_param=confidence_filter).get(1, np.array([])).tolist()
            view_factors(st.session_state.document_content,
                         ids[0:min(len(ids), 15)], colors[1])

        st.write("___")
        st.write("")
        class_id = 2
        annotated_text(
            (
                f"{reversed_mapping[list(output[3].items())[class_id][0]]} - {int(list(output[3].items())[class_id][1] * 100)}%",
                "", colors[class_id])
        )
        view_third_class = st.button("Выделить", key=f"view_third")
        if view_third_class:
            ids = get_confidence_sentences_ids(*output[2], quantile_param=confidence_filter).get(2, np.array([])).tolist()
            view_factors(st.session_state.document_content,
                         ids[0:min(len(ids), 15)], colors[2])

        st.write("___")
        st.write("")
        class_id = 3
        annotated_text(
            (
                f"{reversed_mapping[list(output[3].items())[class_id][0]]} - {int(list(output[3].items())[class_id][1] * 100)}%",
                "", colors[class_id])
        )
        view_fourth_class = st.button("Выделить", key=f"view_fourth")
        if view_fourth_class:
            ids = get_confidence_sentences_ids(*output[2], quantile_param=confidence_filter).get(3, np.array([])).tolist()
            view_factors(st.session_state.document_content,
                         ids[0:min(len(ids), 15)], colors[3])

        st.write("___")
        st.write("")
        class_id = 4
        annotated_text(
            (
                f"{reversed_mapping[list(output[3].items())[class_id][0]]} - {int(list(output[3].items())[class_id][1] * 100)}%",
                "", colors[class_id])
        )
        view_fifth_class = st.button("Выделить", key=f"view_fifth")
        if view_fifth_class:
            ids = get_confidence_sentences_ids(*output[2], quantile_param=confidence_filter).get(4, np.array([])).tolist()
            view_factors(st.session_state.document_content,
                         ids[0:min(len(ids), 15)], colors[4])

        st.write("___")
        st.write("")
        entities = get_entities(st.session_state.document_content)
        entities_loc = []
        entities_org = []
        entities_per = []
        for i in entities.keys():
            if entities[i]['type'] == 'LOC':
                entities_loc.append(i)
            if entities[i]['type'] == 'ORG':
                entities_org.append(i)
            if entities[i]['type'] == 'PER':
                entities_per.append(i)
        print(f'LOC: {entities_loc}')
        print(f'ORG: {entities_org}')
        print(f'PER: {entities_per}')

        if len(entities.keys()) > 0:
            st.subheader("Важные сущности")
        if len(entities_org) > 0:
            st.write("Организации (Погуглить про них?):")
            for organization in entities_org:
                st.write(f"[{organization}](https://www.google.com/search?q={organization.replace(' ', '+')})")

        if len(entities_loc) > 0:
            st.write("Локации (Открыть на карте?):")
            for location in entities_loc:
                st.write(f"[{location}](https://www.google.com/maps/search/{location.replace(' ', '+')})")

        if len(entities_per) > 0:
            st.write("Персоны (Погуглить про них?):")
            for person in entities_per:
                st.write(f"[{person}](https://www.google.com/search?q={person.replace(' ', '+')})")

    with viewing_column:
        confidence_filter = st.session_state.confidence_filter

        st.subheader("Просмотр документа")

        if view_first_class:
            view_document(st.session_state.document_content,
                          get_confidence_sentences_ids(*output[2], quantile_param=confidence_filter)
                          .get(0, np.array([])).tolist(), colors[0])
        elif view_second_class:
            view_document(st.session_state.document_content,
                          get_confidence_sentences_ids(*output[2], quantile_param=confidence_filter)
                          .get(1, np.array([])).tolist(), colors[1])
        elif view_third_class:
            view_document(st.session_state.document_content,
                          get_confidence_sentences_ids(*output[2], quantile_param=confidence_filter)
                          .get(2, np.array([])).tolist(), colors[2])
        elif view_fourth_class:
            view_document(st.session_state.document_content,
                          get_confidence_sentences_ids(*output[2], quantile_param=confidence_filter)
                          .get(3, np.array([])).tolist(), colors[3])
        elif view_fifth_class:
            view_document(st.session_state.document_content,
                          get_confidence_sentences_ids(*output[2], quantile_param=confidence_filter)
                          .get(4, np.array([])).tolist(), colors[4])
except ValueError as err:
    pass
