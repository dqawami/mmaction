from collections import OrderedDict

import numpy as np
from sklearn.metrics import confusion_matrix


def softmax(x, dim=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / e_x.sum(axis=dim, keepdims=True)


def norm_confusion_matrix(scores, labels):
    pred = np.argmax(scores, axis=1)
    cf = confusion_matrix(labels, pred).astype(float)

    cls_cnt = np.sum(cf, axis=1, keepdims=True)
    norm_cm = cf / cls_cnt

    return norm_cm


def invalid_pred_info(scores, labels, k=5, scale=1.0):
    pred = np.argsort(scores, axis=-1)[:, -k:]
    conf = np.max(softmax(scale * scores, dim=-1), axis=1)

    invalid_mask = np.array([labels[i] not in pred[i] for i in range(len(labels))])

    invalid_ids = np.arange(len(pred))[invalid_mask]
    invalid_conf = conf[invalid_mask]

    return invalid_ids, invalid_conf


def mean_class_accuracy(scores, labels):
    pred = np.argmax(scores, axis=1)
    cf = confusion_matrix(labels, pred)

    cls_cnt = cf.sum(axis=1)
    cls_cnt[cls_cnt == 0] = 1

    cls_hit = np.diag(cf)

    return np.mean(cls_hit.astype(float) / cls_cnt.astype(float))


def mean_top_k_accuracy(scores, labels, k=1):
    """MS-ASL like top-k accuracy definition.
    """

    idx = np.argsort(-scores, axis=-1)[:, :k]
    labels = np.array(labels)
    matches = np.any(idx == labels.reshape([-1, 1]), axis=-1)

    classes = np.unique(labels)

    accuracy_values = []
    for class_id in classes:
        mask = labels == class_id
        num_valid = np.sum(mask)
        factor = 1. / float(num_valid) if num_valid > 0 else 1.0
        accuracy_values.append(factor * np.sum(matches[mask]))

    return np.mean(accuracy_values) if len(accuracy_values) > 0 else 1.0


def mean_average_precision(scores, labels):
    def _ap(in_recall, in_precision):
        mrec = np.concatenate((np.zeros([1, in_recall.shape[1]], dtype=np.float32),
                               in_recall,
                               np.ones([1, in_recall.shape[1]], dtype=np.float32)))
        mpre = np.concatenate((np.zeros([1, in_precision.shape[1]], dtype=np.float32),
                               in_precision,
                               np.zeros([1, in_precision.shape[1]], dtype=np.float32)))

        for i in range(mpre.shape[0] - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        all_ap = []
        cond = mrec[1:] != mrec[:-1]
        for k in range(cond.shape[1]):
            i = np.where(cond[:, k])[0]
            all_ap.append(np.sum((mrec[i + 1, k] - mrec[i, k]) * mpre[i + 1, k]))

        return np.array(all_ap, dtype=np.float32)

    one_hot_labels = np.zeros_like(scores, dtype=np.int32)
    one_hot_labels[np.arange(len(labels)), labels] = 1

    idx = np.argsort(-scores, axis=0)
    sorted_labels = np.take_along_axis(one_hot_labels, idx, axis=0)

    matched = sorted_labels == 1

    tp = np.cumsum(matched, axis=0).astype(np.float32)
    fp = np.cumsum(~matched, axis=0).astype(np.float32)

    num_pos = np.sum(one_hot_labels, axis=0)
    num_pos[num_pos == 0] = 1
    num_pos = num_pos.astype(np.float32)

    recall = tp / num_pos.reshape([1, -1])
    precision = tp / (tp + fp)

    ap = _ap(recall, precision)
    mean_ap = np.mean(ap)

    return mean_ap


def top_k_acc(score, lb_set, k=3):
    idx = np.argsort(score)[-k:]
    return len(lb_set.intersection(idx)), len(lb_set)


def top_k_hit(score, lb_set, k=3):
    idx = np.argsort(score)[-k:]
    return len(lb_set.intersection(idx)) > 0, 1


def top_k_accuracy(scores, labels, k=(1,)):
    res = []
    for kk in k:
        hits = []
        for x, y in zip(scores, labels):
            y = [y] if isinstance(y, int) else y
            hits.append(top_k_hit(x, set(y), k=kk)[0])
        res.append(np.mean(hits))
    return res


def invalid_filtered(scores, labels, min_num_fails=1):
    pred = np.argmax(scores, axis=1)
    invalid_mask = pred != labels

    invalid_by_classes = dict()
    for i, label in enumerate(labels):
        if invalid_mask[i]:
            if label not in invalid_by_classes:
                invalid_by_classes[label] = []

            invalid_by_classes[label].append(i)

    out_invalid_classes = {label: ids for label, ids in invalid_by_classes.items() if len(ids) >= min_num_fails}
    out_invalid_classes = OrderedDict(sorted(out_invalid_classes.items(), key=lambda kv: -len(kv[1])))

    return out_invalid_classes
