import numpy as np
import string
import re
from collections import Counter
import re

# Normalize percentage/decimal answers for numerical comparison
def normalize_percentage(value):
    try:
        if "%" in value:
            return str(float(value.replace("%", "").strip()) / 100)
        return str(float(value))
    except ValueError:
        return value


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def accuracy(preds, labels):
    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        if pred == target:
            match_count += 1

    return 100 * (match_count / len(preds))



def numerical_accuracy(prediction, ground_truths, tolerance=1e-5):
    """Check numerical matches within tolerance."""
    try:
        pred_value = float(prediction)
        for gt in ground_truths:
            gt_value = float(gt)
            if abs(pred_value - gt_value) <= tolerance:
                return 1
        return 0
    except ValueError:
        return 0


def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# def find_entity_tags(sentence):
#     entity_regex = r'(.+?)(?=\s<|$)'
#     tag_regex = r'<(.+?)>'
#     entity_names = re.findall(entity_regex, sentence)
#     tags = re.findall(tag_regex, sentence)

#     results = {}
#     for entity, tag in zip(entity_names, tags):
#         if "<" in entity:
#             results[entity.split("> ")[1]] = tag
#         else:
#             results[entity] = tag
#     return results

def match(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0

def evaluate_finqa(prediction, ground_truths, tolerance=1e-5):
    em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
    f1 = metric_max_over_ground_truths(qa_f1_score, prediction, ground_truths)
    num_acc = numerical_accuracy(prediction, ground_truths, tolerance)
    return {"em": em, "f1": f1, "numerical_accuracy": num_acc}
