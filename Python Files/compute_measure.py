import numpy as np



def compute_measure(true_labels, predicted_labels):
    t_idx = (true_labels == predicted_labels)
    f_idx = np.logical_not(t_idx)
    p_idx = (true_labels > 1)
    n_idx = np.sum(np.logical_and(t_idx, p_idx))

    tn = np.sum(np.logical_and(t_idx, n_idx))
    