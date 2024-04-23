import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

predicted_label = np.array( [-1, 1, 1, -1, 1, -1, 1, -1, -1, -1])
true_label = np.array( [ 1, 1, 1, -1, 1, 1, 1, 1, 1, 1 ])

def compute_measure(true_label, predicted_label):
    t_idx = (true_label == predicted_label) # truely predicted
    f_idx = np.logical_not(t_idx) # falsely predicted
    p_idx = (true_label>0) # positive targets
    n_idx = np.logical_not(p_idx) #negative targets
    tp = np.sum(np.logical_and(t_idx, p_idx)) # TP
    tn = np.sum(np.logical_and(t_idx, n_idx)) # TN
    fp = np.sum(n_idx) - tn
    fn = np.sum(p_idx) - tp
    tp_fp_tn_fn_list = []
    tp_fp_tn_fn_list.append(tp)
    tp_fp_tn_fn_list.append(fp)
    tp_fp_tn_fn_list.append(tn)
    tp_fp_tn_fn_list.append(fn)
    tp_fp_tn_fn_list = np.array(tp_fp_tn_fn_list)
    tp = tp_fp_tn_fn_list[0]
    fp = tp_fp_tn_fn_list[1]
    tn = tp_fp_tn_fn_list[2]
    fn = tp_fp_tn_fn_list[3]

    with np.errstate(divide= 'ignore'):
        sen = (1.0 * tp)/(tp + fn)
    with np.errstate(divide= 'ignore'):
        spec = (1.0 * tn)/(tn + fp)
    with np.errstate(divide= 'ignore'):
        ppr = (1.0*tp)/(tp+fp)
    with np.errstate(divide= 'ignore'):
        npr = (1.0*tn)/(tn+fn)
    with np.errstate(divide= 'ignore'):
        f1=tp/(tp+0.5*(fp+fn))
        acc = (tp+tn)*1.0/(tp+fp+tn+fn)
        d = np.log2(1 + acc) + np.log2(1 + (sen+spec) /2)
    ans = []
    ans.append(acc)
    ans.append(sen)
    ans.append(spec)
    ans.append(ppr)
    ans.append(npr)
    ans.append(f1)
    ans.append(d)
    return ans

def model_metrics(true,pred):
    ans=compute_measure(true,pred)
    print("Accuracy is {0:4f}".format(ans[0]))
    print("Sensitivity is {0:4f}".format(ans[1]))
    print("Specificity is {0:4f}".format(ans[2]))
    print("Precision is {0:4f}".format(ans[3]))
    print("Negative Prediction ratio is{0:3f}".format(ans[4]))
    print("F1-score is {0:3f}".format(ans[5]))
    print("Diagnostic index is {0:4f}".format(ans[6]))
    print("\n")
    print(classification_report(true,pred))




model_metrics(true_label,predicted_label)
conf_matrix = confusion_matrix(true_label,predicted_label)
print(conf_matrix)

plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
    xticklabels=['-1', '+1'],
    yticklabels=['-1', '+1'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()