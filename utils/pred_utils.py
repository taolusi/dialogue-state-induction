import copy
import collections
import json
from functools import reduce

from fuzzywuzzy import fuzz


def get_predicted_dialogs(predictions, dialog_paths, threshold):
    """Mapping predicted slots (some numbers) to reference labels."""
    dialogs = []
    for dialog_path in dialog_paths:
        annotation_path = dialog_path.with_suffix(".annotation")
        with annotation_path.open("r") as f:
            dialog = json.load(f)
            dialogs.extend(dialog)

    # Collect sores of reference labels corresponding to each predicted slot.
    scores = collections.defaultdict(
        lambda: collections.defaultdict(int))

    predictions = iter(predictions)
    # Turn level predictions correspond to states update, joint predictions
    # correspond to typical dialogue states.
    turn_predictions = []
    joint_predictions = []

    for dialog in dialogs:
        # Each prediction is stored in a dict.
        prev_prediction = {}
        for turn in dialog['turns']:
            prediction = next(predictions)
            check(turn, prediction)
            reference = get_states(turn["states"])
            prediction = filter_predictions(
                turn, prediction, threshold)
            turn_predictions.append(copy.deepcopy(prediction))
            prediction.update(prev_prediction)
            joint_predictions.append(prediction)
            greedy_map(reference, prediction, scores)
            prev_prediction = prediction

    # Slot map is dict in which keys are predicted slot numbers and
    # values are mapped reference slots.
    slot_map = {}
    for pred_slot, score in scores.items():
        ref_slot = sorted(
            score.items(),
            key=lambda x: x[1],
            reverse=True)[0]
        if ref_slot[1] == 0:
            slot_map[pred_slot] = 'none'
        else:
            # Reference slot with most matches is mapped to the predicted slot.
            slot_map[pred_slot] = ref_slot[0]

    mapped_turn_predictions = []
    mapped_joint_predictions = []
    for turn_prediction, joint_prediction in zip(
            turn_predictions, joint_predictions):
        mapped_turn_predictions.append(
            map_predictions(turn_prediction, slot_map))
        mapped_joint_predictions.append(
            map_predictions(joint_prediction, slot_map))

    return mapped_turn_predictions, mapped_joint_predictions


def check(turn, answer):
    candidates_in = reduce(
        lambda x,
        y: list(x) + list(y),
        turn['candidates'].values())
    candidates_in = list(map(lambda x: x[-1], candidates_in))
    candidates_out = list(
        map(lambda x: x['word'], answer))
    assert candidates_in == candidates_out, 'Cannot match in input corpus and output answer!'


def filter_predictions(turn, prediction, threshold):
    """Some predictions are duplicated in slots or values. Those predictions with
    larger probabilities will be retained."""
    candidates = reduce(
        lambda x,
        y: list(x) + list(y),
        turn['candidates'].values())
    filtered_prediction = {}
    assert len(candidates) == len(prediction)
    for i, (candidate, slot_value) in enumerate(zip(candidates, prediction)):
        assert candidate[-1] == slot_value['word']
        if slot_value['prob'][slot_value['slot']] > threshold:
            slot = str(slot_value['domain']) + '-' + str(slot_value['slot'])
            prob = slot_value['prob'][slot_value['slot']]
            value = slot_value['word']
            flag = True
            for j, pred in enumerate(prediction):
                if i != j and (value in pred['word'] or pred['word']
                               in value) and prob < pred['prob'][pred['slot']]:
                    flag = False
            if flag:
                try:
                    prev_prob = filtered_prediction[slot][1]
                except KeyError:
                    filtered_prediction[slot] = [candidate[-1], prob]
                else:
                    if prob > prev_prob:
                        filtered_prediction[slot] = [candidate[-1], prob]
    return filtered_prediction


def get_states(states: dict):
    """All slot value pairs in different domains are stored in a single dict."""
    reference = {}
    for domain, frame in states.items():
        for slot, values in frame['slot_values'].items():
            if slot != "requested_slots":
                reference[slot] = values
    return reference


def get_states_update(states: dict):
    """All slot value pairs in different domains are stored in a single dict."""
    reference = {}
    for domain, frame in states.items():
        for slot, values in frame.items():
            if slot != "requested_slots":
                reference[slot] = values
    return reference


def merge_predictions(prev_predictions, predictions):
    """Return the union of preceding prediction and current prediction. When
    there are multiple values for one slot, latest value will be used"""
    temp = []
    for prev in prev_predictions:
        flag = True
        for cur in predictions:
            if prev[0] == cur[0]:
                flag = False
                break
        if flag:
            temp.append(prev)
    predictions.extend(temp)


def greedy_map(reference, prediction, scores):
    """If predicted value is similar to reference value, then predicted
    the score of reference slot corresponding to the predicted slot will
    be +1."""
    for pred_slot, pred_value in prediction.items():
        for ref_slot, ref_value in reference.items():
            for v in ref_value:
                # A fuzzy matching scenario is applied with a threshold of 80.
                if fuzz.token_sort_ratio(
                        v.lower(), pred_value[0].lower()) > 80:
                    scores[pred_slot][ref_slot] += 1
                    break


def map_predictions(prediction, slot_map):
    """Map predicted numbers to reference slots according to slot map."""
    mapped_prediction = {}
    for slot, value in prediction.items():
        try:
            mapped_slot = slot_map[slot]
        except KeyError:
            pass
        else:
            if mapped_slot != 'none':
                mapped_prediction[mapped_slot] = value[0]
    return mapped_prediction
