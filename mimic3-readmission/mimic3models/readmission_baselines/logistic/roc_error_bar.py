from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from mimic3benchmark.util import *
import os
from scipy import interp

from sklearn.metrics import roc_curve, auc


def read_result(subject_path,file):
    result = dataframe_from_csv(os.path.join(subject_path, file), index_col=None)
    pred=result['prediction'].values.tolist()
    label=result['y_true'].values.tolist()
    data=(pred, label)
    return data


fig = plt.figure(figsize=(7,7))

o_path='/Users/jeffrey0925/Desktop/'

folders=['RF','LR','LSTM_no_embedding','LSTM _Demographic','LSTM_CNN_D_tune']




'''
file1='rf.csv'
file2='svm.csv'
file3='noicd_3_k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch29.test0.597079065251462.state.csv'
file4='demo_4_k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch30.test0.5354190086797693.state.csv'
file5='best_4_k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch13.test0.5961579322814942.state.csv'

files=[file1,file2,file3,file4,file5]

file_names={file1:'RF',
    file2:'LR',
    file3:'LSTM (L48-h CE)',
    file4:'LSTM (L48-h CE + ICD9 +D)',
    file5:'LSTM+CNN (L48-h CE + ICD9 + D)'}
'''
linestyles = ['-', '--', '-.', ':','-']
line=0
file_names=['RF','LR','LSTM (L48-h CE)','LSTM (L48-h CE + ICD9 +D)','LSTM+CNN (L48-h CE + ICD9 + D)']
i = 0
name=0
for folder in folders:
    path=os.path.join(o_path, folder)
    files = list(os.listdir(path))
    print(folder, files)

    mean_fpr = np.linspace(0, 1, 100)

    tprs = []
    aucs = []


    for file in files:
        result=read_result(path, file)
        pred =result[0]
        label =result[1]
        #file_name=file_names[file]
        fpr, tpr, thresh = metrics.roc_curve(label, pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=2, alpha=0.3,label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))

        #auc = metrics.roc_auc_score(label, pred)
        #plt.plot(fpr,tpr,linestyle=linestyles[i],lw=3,label=file_name+"= %0.3f" % auc)
        #plt.plot(fpr,tpr,linestyle=linestyles[i],lw=3,label='fold '+str(i+1)+'= %0.3f' % auc)
        i+=1



    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    file_name=file_names[name]
    name+=1
    plt.plot(mean_fpr, mean_tpr,
         label=file_name+' '+r'(AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),linestyle=linestyles[line],
         lw=3)
    line+=1
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper,  alpha=.3)
    #plt.fill_between(mean_fpr, tprs_lower, tprs_upper,  alpha=.3,label=r'$\pm$ 1 std. dev.')


plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)

plt.xlim([0., 1.])
plt.ylim([0., 1.])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


#plt.plot([0, 1], [0, 1], linestyle='--',lw=2, color='k')



fig.savefig('/Users/jeffrey0925/Downloads/mimic3-benchmarks-master/mimic3models/readmission3/logistic/error_bar222.png')

plt.show()
