import numpy as np
import argparse
import os

from mimic3benchmark.util import *
from collections import Counter

from mimic3benchmark.readers import ReadmissionReader

from mimic3models.preprocessing import Discretizer

from mimic3models import common_utils


from utilities.data_loader import get_embeddings
from sklearn import linear_model, metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import json
from mimic3models.metrics import print_metrics_binary
from mimic3models.readmission_baselines.utils import save_results
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


fig = plt.figure(figsize=(7,7))


def read_diagnose(subject_path,icustay):
    diagnoses = dataframe_from_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)
    diagnoses=diagnoses.ix[(diagnoses.ICUSTAY_ID==int(icustay))]
    diagnoses=diagnoses['ICD9_CODE'].values.tolist()

    return diagnoses

def get_diseases(names,path):
    disease_list=[]
    namelist=[]
    for element in names:
        x=element.split('_')
        namelist.append((x[0],x[1]))
    for x in namelist:
        subject=x[0]
        icustay=x[1]
        subject_path=os.path.join(path, subject)
        disease = read_diagnose(subject_path,icustay)
        disease_list.append(disease)
    return disease_list


def disease_embedding(embeddings, word_indices,diseases_list):
    emb_list=[]
    for diseases in diseases_list:
        emb_period=[0]*300
        skip=0
        for disease in diseases:
            k='IDX_'+str(disease)
            if k not in word_indices.keys():
                skip+=1
                continue
            index=word_indices[k]
            emb_disease=embeddings[index]
            emb_period = [sum(x) for x in zip(emb_period, emb_disease)]
        emb_list.append(emb_period)
    return emb_list

def find_majority(k):
    myMap = {}
    maximum = ( '', 0 )
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum

def logit(X,cont_channels,begin_pos, end_pos):

    X=np.asmatrix(X)

    index=list(range(59))
    no=list(range(59,76))
    idx_features=[]
    features=[]
    majority_index=list(set(index)-set(cont_channels))
    reg_index=list(set(cont_channels)-set(no))


    for idx in majority_index:
        begine=0
        end=0
        for i, item in enumerate(begin_pos):
            if item==idx:
                begin=idx
                end=end_pos[i]

                flat_list = [ map(int, my_lst) for my_lst in X[:,begin:end].tolist()]

                flat_list = [ (''.join(map(str, my_lst))) for my_lst in flat_list]

                value=find_majority(flat_list)[0]
                for ch in list(value):
                    idx_features.append(float(ch))
    for idx in reg_index:
        regr = linear_model.LinearRegression()
        flat_list = [item for sublist in X[:,idx].tolist() for item in sublist]

        time = [[i] for i in range(1,len(flat_list)+1)]

        regr.fit(time, flat_list)
        a=regr.coef_[0]
        b=regr.intercept_
        features.append(a)
        features.append(b)
    return idx_features, features

def column_sum(M):
    s=M.shape[0]
    column_sums = [sum([row[i] for row in M]) for i in range(0, len(M[0]))]
    newList = [x /s for x in column_sums]
    return newList


embeddings, word_indices = get_embeddings(corpus='claims_codes_hs', dim=300)

# Build readers, discretizers, normalizers
train_reader = ReadmissionReader(dataset_dir='/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/data/',
                                         listfile='/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/0_train_listfile801010.csv')

val_reader = ReadmissionReader(dataset_dir='/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/data/',
                                       listfile='/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/0_val_listfile801010.csv')

test_reader = ReadmissionReader(dataset_dir='/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/data/',
                                    listfile='/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/0_test_listfile801010.csv')


discretizer = Discretizer(timestep=float(1.0),
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

N=train_reader.get_number_of_examples()
ret = common_utils.read_chunk(train_reader, N)
data = ret["X"]
ts = ret["t"]
train_y = ret["y"]
train_names = ret["name"]
diseases_list=get_diseases(train_names, '/Users/jeffrey0925/MIMIC-III-clean/data/')
diseases_embedding=disease_embedding(embeddings, word_indices,diseases_list)

d, discretizer_header, begin_pos, end_pos = discretizer.transform_reg(data[0])

discretizer_header=discretizer_header.split(',')



cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]


da= [discretizer.transform_end_t_hours_reg(X, los=t)[1] for (X, t) in zip(data, ts)]
mask=[column_sum(x) for x in da]

#train_set=[]
d= [discretizer.transform_end_t_hours_reg(X, los=t)[0] for (X, t) in zip(data, ts)]

idx_features_train= [logit(X,cont_channels,begin_pos, end_pos)[0] for X in d]
features_train = [logit(X,cont_channels,begin_pos, end_pos)[1] for X in d]

train_X=features_train

scaler = MinMaxScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)

train_X = [np.hstack([X, d]) for (X, d) in zip(train_X, diseases_embedding)]


train_X = [np.hstack([X, d]) for (X, d) in zip(train_X, idx_features_train)]

train_X = [np.hstack([X, d]) for (X, d) in zip(train_X, mask)]

labels_1 = []
labels_0 = []
data_1 = []
data_0 = []
for i in range(len(train_y)):
    if train_y[i] == 1:
        labels_1.append(train_y[i])
        data_1.append(train_X[i])
    elif train_y[i] == 0:
        labels_0.append(train_y[i])
        data_0.append(train_X[i])



print('labels_1:', len(labels_1))
print('labels_0:', len(labels_0))
indices = np.random.choice(len(labels_0), len(labels_1),replace=False)
labels_0_sample =[labels_0[idx] for idx in indices]
#print('labels_0_sample: ', labels_0_sample)
print('len(labels_0_sample): ', len(labels_0_sample))

data_0_sample =[data_0[idx] for idx in indices]
#print('data_0_sample: ', data_0_sample)
print('len(data_0_sample): ', len(data_0_sample))

data_new=data_0_sample+data_1
label_new=labels_0_sample+labels_1

c = list(zip(data_new, label_new))

random.shuffle(c)

data_new, label_new = zip(*c)
train_X=list(data_new)
train_y=list(label_new)
#print('data_new: ', data_new)
print('data_new: ', len(train_X))
#print('label_new: ', label_new)
print('label_new: ', len(train_y))

#-------------------------

N_val=val_reader.get_number_of_examples()
ret_val = common_utils.read_chunk(val_reader, N_val)
data_val = ret_val["X"]
ts_val = ret_val["t"]
val_y= ret_val["y"]
val_names = ret_val["name"]

diseases_list_val=get_diseases(val_names, '/Users/jeffrey0925/MIMIC-III-clean/data/')
diseases_embedding_val=disease_embedding(embeddings, word_indices,diseases_list_val)


#----------
da_val= [discretizer.transform_end_t_hours_reg(X, los=t)[1] for (X, t) in zip(data_val, ts_val)]
mask_val=[column_sum(x) for x in da_val]

#train_set=[]
d_val= [discretizer.transform_end_t_hours_reg(X, los=t)[0] for (X, t) in zip(data_val, ts_val)]

#---------
#val_set=[]
idx_features_val = [logit(X,cont_channels,begin_pos, end_pos)[0] for X in d_val]
features_val = [logit(X,cont_channels,begin_pos, end_pos)[1] for X in d_val]

val_X = scaler.transform(features_val)

val_X = [np.hstack([X, d]) for (X, d) in zip(val_X, diseases_embedding_val)]
val_X = [np.hstack([X, d]) for (X, d) in zip(val_X, idx_features_val)]
val_X = [np.hstack([X, d]) for (X, d) in zip(val_X, mask_val)]


#-------------------------

N_test=test_reader.get_number_of_examples()
ret_test = common_utils.read_chunk(test_reader, N_test)
data_test = ret_test["X"]
ts_test = ret_test["t"]
test_y= ret_test["y"]
test_names = ret_test["name"]

diseases_list_test = get_diseases(test_names, '/Users/jeffrey0925/MIMIC-III-clean/data/')
diseases_embedding_test=disease_embedding(embeddings, word_indices,diseases_list_test)

#----------
da_test= [discretizer.transform_end_t_hours_reg(X, los=t)[1] for (X, t) in zip(data_test, ts_test)]
mask_test=[column_sum(x) for x in da_test]

#train_set=[]
d_test= [discretizer.transform_end_t_hours_reg(X, los=t)[0] for (X, t) in zip(data_test, ts_test)]
#----------

#data_test=[]
idx_features_test = [logit(X,cont_channels,begin_pos, end_pos)[0] for X in d_test]
features_test = [logit(X,cont_channels,begin_pos, end_pos)[1] for X in d_test]

test_X = scaler.transform(features_test)

test_X = [np.hstack([X, d]) for (X, d) in zip(test_X, diseases_embedding_test)]
test_X = [np.hstack([X, d]) for (X, d) in zip(test_X, idx_features_test)]
test_X = [np.hstack([X, d]) for (X, d) in zip(test_X, mask_test)]

#=========SVM====================
penalty = ('l2')
#file_name = '{}.{}.{}.C{}'.format(penalty, 0.001)


logreg = SVC(probability=True)
logreg.fit(train_X, train_y)

#-----------------
common_utils.create_directory('svm_results')
common_utils.create_directory('svm_predictions')


with open(os.path.join('svm_results', 'train.json'), 'w') as res_file:
    ret = print_metrics_binary(train_y, logreg.predict_proba(train_X))
    ret = {k : float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

with open(os.path.join('svm_results', 'val.json'), 'w') as res_file:
    ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

prediction = logreg.predict_proba(test_X)[:, 1]

with open(os.path.join('svm_results', 'test.json'), 'w') as res_file:
    ret = print_metrics_binary(test_y, prediction)
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

    predictions = np.array(prediction)
    if (len(predictions.shape) == 1):
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    fpr, tpr, thresh = metrics.roc_curve(test_y, predictions[:, 1])
    auc = metrics.roc_auc_score(test_y, predictions[:, 1])
    plt.plot(fpr,tpr,lw=2,label="SVM= %0.3f" % auc)

save_results(test_names, prediction, test_y, os.path.join('svm_predictions', 'svm.csv'))


#=============LR================


logreg = LogisticRegression(penalty=penalty, C=0.001, random_state=42)
logreg.fit(train_X, train_y)

#-----------------
common_utils.create_directory('lr_results')
common_utils.create_directory('lr_predictions')

with open(os.path.join('lr_results', 'train.json'), 'w') as res_file:
    ret = print_metrics_binary(train_y, logreg.predict_proba(train_X))
    ret = {k : float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

with open(os.path.join('lr_results', 'val.json'), 'w') as res_file:
    ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

prediction = logreg.predict_proba(test_X)[:, 1]

with open(os.path.join('lr_results', 'test.json'), 'w') as res_file:
    ret = print_metrics_binary(test_y, prediction)
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

    predictions = np.array(prediction)
    if (len(predictions.shape) == 1):
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    fpr, tpr, thresh = metrics.roc_curve(test_y, predictions[:, 1])
    auc = metrics.roc_auc_score(test_y, predictions[:, 1])
    plt.plot(fpr,tpr,lw=2,label="LR= %0.3f" % auc)

save_results(test_names, prediction, test_y, os.path.join('lr_predictions', 'lr.csv'))

#===========RF==================


logreg = RandomForestClassifier(oob_score=True, max_depth=50, random_state=0)
logreg.fit(train_X, train_y)

#-----------------
common_utils.create_directory('rf_results')
common_utils.create_directory('rf_predictions')


with open(os.path.join('rf_results', 'train.json'), 'w') as res_file:
    ret = print_metrics_binary(train_y, logreg.predict_proba(train_X))
    ret = {k : float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

with open(os.path.join('rf_results', 'val.json'), 'w') as res_file:
    ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

prediction = logreg.predict_proba(test_X)[:, 1]

with open(os.path.join('rf_results', 'test.json'), 'w') as res_file:
    ret = print_metrics_binary(test_y, prediction)
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

    predictions = np.array(prediction)
    if (len(predictions.shape) == 1):
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    fpr, tpr, thresh = metrics.roc_curve(test_y, predictions[:, 1])
    auc = metrics.roc_auc_score(test_y, predictions[:, 1])
    plt.plot(fpr,tpr,lw=2,label="RF= %0.3f" % auc)

save_results(test_names, prediction, test_y, os.path.join('rf_predictions', 'rf.csv'))


#=============NB================


logreg = GaussianNB()
logreg.fit(train_X, train_y)

#-----------------
common_utils.create_directory('nb_results')
common_utils.create_directory('nb_predictions')


with open(os.path.join('nb_results', 'train.json'), 'w') as res_file:
    ret = print_metrics_binary(train_y, logreg.predict_proba(train_X))
    ret = {k : float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

with open(os.path.join('nb_results', 'val.json'), 'w') as res_file:
    ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

prediction = logreg.predict_proba(test_X)[:, 1]

with open(os.path.join('nb_results', 'test.json'), 'w') as res_file:
    ret = print_metrics_binary(test_y, prediction)
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

    predictions = np.array(prediction)
    if (len(predictions.shape) == 1):
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    fpr, tpr, thresh = metrics.roc_curve(test_y, predictions[:, 1])
    auc = metrics.roc_auc_score(test_y, predictions[:, 1])
    plt.plot(fpr,tpr,lw=2,label="NB= %0.3f" % auc)

save_results(test_names, prediction, test_y, os.path.join('nb_predictions', 'nb.csv'))


#=============================
plt.plot([0, 1], [0, 1], linestyle='--',lw=2, color='k')

plt.xlim([0., 1.])
plt.ylim([0., 1.])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")



fig.savefig('/Users/jeffrey0925/Downloads/mimic3-benchmarks-master/mimic3models/readmission3/logistic_cv_0/ROC0.png')

plt.show()
