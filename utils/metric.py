import numpy as np
from dataclasses import dataclass

from fuzzywuzzy import fuzz


@dataclass()
class Metric():
    acc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    def update(self, acc, f1, precision, recall):
        self.acc += acc
        self.f1 += f1
        self.precision += precision
        self.recall += recall

    def compute_scores(self, total):
        self.acc_score = self.acc / float(total) if total != 0 else 0
        self.f1_score = self.f1 / float(total) if total != 0 else 0
        self.precision_score = self.precision / float(total) if total != 0 else 0
        self.recall_score = self.recall / float(total) if total != 0 else 0


def compute_acc(slot_values_ref, slot_values_hyp):
    """ACC is computed on all slot value pairs. Following DSTC8 SGD dataset, a fuzzy matching scenario
    is applied, which means ACC is a score in range [0.0, 1.0] instead of a bool value."""

    # Return 1.0 if all slots and values are exactly matched.
    if slot_values_hyp == slot_values_ref:
        return 1.0
    # Return 0.0 if hypothesis keys are different from reference keys.
    elif slot_values_ref.keys() != slot_values_hyp.keys():
        return 0.0
    else:
        # For each slot, there is a score of reference value and hypothesis value. Average score for
        # all slots will be returned finally.
        list_cor = []
        for slot, value_ref_list in slot_values_ref.items():
            value_hyp = slot_values_hyp[slot]
            cor = value_match(value_ref_list, value_hyp)
            list_cor.append(cor)
        return np.mean(list_cor)


def compute_prf(slot_values_ref, slot_values_hyp):
    """Like ACC, PRF is also measured using a fuzzy matching scenario."""
    TP, FP, FN = 0, 0, 0
    if len(slot_values_ref.items()) != 0:
        for slot, value_ref_list in slot_values_ref.items():
            try:
                value_hyp = slot_values_hyp[slot]
            except KeyError:
                FN += 1
            else:
                cor = value_match(value_ref_list, value_hyp)
                # True positives are correctness instead of bool values.
                TP += cor

        for slot, value_hyp in slot_values_hyp.items():
            if slot not in slot_values_ref:
                FP += 1

        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        if len(slot_values_hyp.items()) == 0:
            precision, recall, F1 = 1, 1, 1
        else:
            precision, recall, F1 = 0, 0, 0

    return F1, precision, recall


def value_match(str_ref_list, str_hyp, use_fuzzy_match=True):
    """Calculate slot values correctness.

    Args:
      str_ref_list: a list of reference strings.
      str_hyp: the hypothesis string.
      use_fuzzy_match: whether to use fuzzy string matching.

    Returns:
      score: The highest fuzzy string match score of the references and hypotheis.
    """
    score = 0.0
    for str_ref in str_ref_list:
        if not use_fuzzy_match:
            match_score = float(str_ref == str_hyp)
        else:
            match_score = fuzzy_string_match(str_ref, str_hyp)
        score = max(score, match_score)
    return score


def fuzzy_string_match(str_ref, str_hyp):
    """Returns fuzzy string similarity score in range [0.0, 1.0]."""

    # The higher the score, the higher the similarity between the two strings.
    return fuzz.token_sort_ratio(str_ref.lower(), str_hyp.lower()) / 100.0
