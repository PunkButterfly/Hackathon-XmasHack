import numpy as np


def get_confidence_sentences_ids(flat_predictions, predicted_labels, most_confident_labels, quantile_param=0.75):
    most_dominant_sequences = {}
    for label in most_confident_labels[1]:
        fixed_label_probs = flat_predictions[np.where(predicted_labels == label)]
        mean = np.quantile(fixed_label_probs[:, label], quantile_param)
        dominant_ids = np.where(fixed_label_probs[:, label] > mean)
        most_dominant_indices = np.where(predicted_labels == label)[0][dominant_ids]

        most_dominant_sequences[label] = most_dominant_indices

    return most_dominant_sequences
