from mimic3models import common_utils
import numpy as np
import os
from mimic3models import nn_utils


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

    data = nn_utils.pad_zeros_from_left(data)

    whole_data = (data, labels)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))
