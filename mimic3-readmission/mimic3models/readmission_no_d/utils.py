from mimic3models import common_utils
import numpy as np
import os
from mimic3models import nn_utils
import random


def load_data(reader, discretizer, normalizer, diseases_embedding, return_names=False):
    N = reader.get_number_of_examples()

    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform_end_t_hours(X, los=t)[0] for (X, t) in zip(data, ts)]



    if (normalizer is not None):
        data = [normalizer.transform(X) for X in data]


    data = [np.hstack([X, [d]*len(X)]) for (X, d) in zip(data, diseases_embedding)]

    data = nn_utils.pad_zeros(data)

    whole_data = (data, labels)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}
#================

def load_train_data(reader, discretizer, normalizer, diseases_embedding, return_names=False):
    N = reader.get_number_of_examples()

    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform_end_t_hours(X, los=t)[0] for (X, t) in zip(data, ts)]


    if (normalizer is not None):
        data = [normalizer.transform(X) for X in data]
    data = [np.hstack([X, [d]*len(X)]) for (X, d) in zip(data, diseases_embedding)]
    labels_1=[]
    labels_0=[]
    data_1=[]
    data_0=[]
    for i in range(len(labels)):
        if labels[i]==1:
            labels_1.append(labels[i])
            data_1.append(data[i])
        elif labels[i] == 0:
            labels_0.append(labels[i])
            data_0.append(data[i])

    print('labels_1:', len(labels_1))
    print('labels_0:', len(labels_0))
    indices = np.random.choice(len(labels_0), len(labels_1),replace=False)
    labels_0_sample =[labels_0[idx] for idx in indices]
    print('len(labels_0_sample): ', len(labels_0_sample))

    data_0_sample =[data_0[idx] for idx in indices]
    print('len(data_0_sample): ', len(data_0_sample))

    data_new=data_0_sample+data_1
    label_new=labels_0_sample+labels_1

    c = list(zip(data_new, label_new))

    random.shuffle(c)

    data_new, label_new = zip(*c)
    data_new=list(data_new)
    label_new=list(label_new)
    print('data_new: ', len(data_new))
    print('label_new: ', len(label_new))

    data = nn_utils.pad_zeros(data_new)

    whole_data = (data, label_new)
    if not return_names:
        return whole_data
    return {"data": whole_data}
#================
def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))