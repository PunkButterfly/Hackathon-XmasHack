import pymorphy2
import numpy as np
from natasha import NewsNERTagger, NewsEmbedding, Doc


def get_confidence_sentences_ids(flat_predictions, predicted_labels, most_confident_labels, quantile_param=0.75):
    most_dominant_sequences = {}
    for label in most_confident_labels[1]:
        fixed_label_probs = flat_predictions[np.where(predicted_labels == label)]
        mean = np.quantile(fixed_label_probs[:, label], quantile_param)
        dominant_ids = np.where(fixed_label_probs[:, label] > mean)
        most_dominant_indices = np.where(predicted_labels == label)[0][dominant_ids]

        most_dominant_sequences[label] = most_dominant_indices

    return most_dominant_sequences


def get_entities(text):
    """
    Input: string

    Ouput:

    dict = {
      word: {'start_stop': (start, stop), 'type': type}
      ...
      }
    ```
    """
    emb = NewsEmbedding()
    ner_tagger = NewsNERTagger(emb)

    doc = Doc(text)
    try:
        doc.tag_ner(ner_tagger)
    except:
        pass

    morph = pymorphy2.MorphAnalyzer()

    dic = dict()
    stop_words = []

    # производим обработку полученных именованных сущностей
    for span in doc.spans:
        if span.text[-1] == ' ' or span.text[-1] == '(' or span.text[-1] == ')':
            continue
        elif span.type == "ORG":
            if not (span.text[0].isupper() and span.text[1].isupper()):
                continue
            elif len(span.text) > 4 and span.text[-1].isupper():
                continue
        elif span.text[0].isupper() and span.text[-1].isupper() and len(span.text) > 3:
            continue

        word = morph.parse(span.text)[0].normal_form

        if (span.text not in dic) and (span.text not in stop_words):
            dic[word] = {'start_stop': [(span.start, span.stop)], 'type': span.type}
        else:
            if dic[word]['type'] != span.type:
                stop_words.append(word)
                dic.pop(word)
            else:
                dic[word]['start_stop'].append((span.start, span.stop))

    return dic
