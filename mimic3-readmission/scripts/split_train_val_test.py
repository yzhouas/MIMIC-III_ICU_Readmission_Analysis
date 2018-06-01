import os
import shutil
import argparse
from sklearn.model_selection import KFold
header='stay,period_length,y_true'

patients = set()
with open("/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/data/listfile.csv", "r") as valset_file:
    for line in valset_file:
        x= line.split(',')
        z=x[0].split('_')
        patients.add(z[0])
patients=list(patients)
with open("/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/data/listfile.csv", "r") as listfile:
    lines = listfile.readlines()

k_fold =KFold(n_splits=10,shuffle=True)
folds=[]
for train_indices, test_indices in k_fold.split(patients):
    print('train_index:%s , test_index: %s ' % (train_indices, test_indices))
    folds.append(test_indices)
cvs=[[0,1,2,3,4,5,6,7,8,9],[2,3,0,1,4,5,6,7,8,9],[4,5,0,1,2,3,6,7,8,9],[6,7,0,1,2,3,4,5,8,9],[8,9,0,1,2,3,4,5,6,7]]
for id, cv in enumerate(cvs):
    train_lines=[]
    for idx, f in enumerate(cv):
        if idx ==0:
            fold=folds[f]
            pa=[patients[x] for x in fold]
            test_lines = [x for x in lines if x[:x.find("_")] in pa]
        elif idx==1:
            fold=folds[f]
            pa=[patients[x] for x in fold]
            val_lines = [x for x in lines if x[:x.find("_")] in pa]
        else:
            fold=folds[f]
            pa=[patients[x] for x in fold]

            train_lines += [x for x in lines if x[:x.find("_")] in pa]

    print(len(train_lines) , len(val_lines) ,len(test_lines),len(lines))
    assert len(train_lines) + len(val_lines) + len(test_lines)== len(lines)

    with open("/Users/jeffrey0925/MIMIC-III-clean/"+str(id)+"_train_listfile801010.csv" , "w") as train_listfile:
        train_listfile.write(header)
        for line in train_lines:
            train_listfile.write(line)

    with open("/Users/jeffrey0925/MIMIC-III-clean/"+str(id)+"_val_listfile801010.csv", "w") as val_listfile:
        val_listfile.write(header)
        for line in val_lines:
            val_listfile.write(line)

    with open("/Users/jeffrey0925/MIMIC-III-clean/"+str(id)+"_test_listfile801010.csv" , "w") as test_listfile:
        test_listfile.write(header)
        for line in test_lines:
            test_listfile.write(line)

