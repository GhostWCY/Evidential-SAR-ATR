import math
import numpy as np
# Some keys used for the following dictionaries
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0

def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + \
            (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / \
                float(bin_dict[binn][COUNT])
    return bin_dict

def expected_calibration_error(confs, preds, labels, num_bins=10):
    """Constructs an expected calibration error metric.
    Args:
      num_bins: Number of bins to maintain over the interval [0, 1].
    """
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * \
            abs(bin_accuracy - bin_confidence)
    return ece

# log likelihood
def nll(y_true, preds):
    """
      Multi-class negative log likelihood.
  If the true label is k, while the predicted vector of probabilities is
  [p_1, ..., p_K], then the negative log likelihood is -log(p_k).

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array
        Ground truth (correct) labels.
    preds : predicted_probs. size (n_samples, c_classes) nd array-like
        Normalized logits, or predicted probabilities for all samples.
    Returns
    -------
    score : float
    """
    preds_target = preds[np.arange(len(y_true)), y_true]#get the predition probability of the correct one
    return np.log(1e-12 + preds_target).mean()

# brier
def brier(y_true, preds):
    """
    Compute brier score(MSE is the statistical and continuous name, Bier Score is the ML and categorical name.
     MSE is the same as brier score )

  If the true label is k, while the predicted vector of probabilities is
  [y_1, ..., y_n], then the Brier score is equal to
    \sum_{i != k} y_i^2 + (y_k - 1)^2.


    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        True labels or binary label indicators. The binary and multiclass cases
        expect labels with shape (n_samples,) while the multilabel case expects
        binary label indicators with shape (n_samples, n_classes).
    preds : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores.

    Returns
    -------
    auc : float
    """
    one_hot_targets = np.zeros(preds.shape)
    one_hot_targets[np.arange(len(y_true)), y_true] = 1.0

    return np.mean(np.sum((preds - one_hot_targets) ** 2, axis=1))
