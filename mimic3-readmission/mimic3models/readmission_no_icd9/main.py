import numpy as np
import argparse
import os
os.environ["KERAS_BACKEND"]="tensorflow"

import importlib.machinery
import re



from mimic3models.readmission_no_icd9 import utils
from mimic3benchmark.readers import ReadmissionReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam



parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
args = parser.parse_args()
print (args)



target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')


# Build readers, discretizers, normalizers

train_reader = ReadmissionReader(dataset_dir='/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/data/',
                                         listfile='/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/0_train_listfile801010.csv')

val_reader = ReadmissionReader(dataset_dir='/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/data/',
                                       listfile='/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/0_val_listfile801010.csv')




discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

N=train_reader.get_number_of_examples()
ret = common_utils.read_chunk(train_reader, N)
data = ret["X"]
ts = ret["t"]



discretizer_header = discretizer.transform(ret["X"][0])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
normalizer = Normalizer(fields=cont_channels)  # choose here onlycont vs all




data = [discretizer.transform_end_t_hours(X, los=t)[0] for (X, t) in zip(data, ts)]



[normalizer._feed_data(x=X) for X in data]
normalizer._use_params()




args_dict = dict(args._get_kwargs())

args_dict['task'] = 'ihm'
args_dict['target_repl'] = target_repl

# Build the model
print ("==> using model {}".format(args.network))
print('os.path.basename(args.network), args.network: ', os.path.basename(args.network), args.network)
model_module = importlib.machinery.SourceFileLoader(os.path.basename(args.network), args.network).load_module()
model = model_module.Network(**args_dict)
suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")
model.final_name = args.prefix + model.say_name() + suffix
print ("==> model.final_name:", model.final_name)


# Compile the model
print ("==> compiling the model")

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).

loss = 'binary_crossentropy'
loss_weights = None
print(model)

model.compile(optimizer=Adam(lr=0.001, beta_1=0.9), loss=loss,loss_weights=loss_weights)

model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


# Read data
train_raw = utils.load_train_data(train_reader, discretizer, normalizer)

print('train_raw: ', len(train_raw[0]))

print('train_raw train_raw[0][0]: ', len(train_raw[0][0]))
print('train_raw train_raw[0][1]: ', len(train_raw[0][1]))

print('train_raw: ', len(train_raw[0][0][0]))


val_raw = utils.load_data(val_reader, discretizer, normalizer)

if target_repl:
    T = train_raw[0][0].shape[0]
    #print('T: ', T)

    def extend_labels(data):
        data = list(data)
        labels = np.array(data[1]) # (B,)
        data[1] = [labels, None]
        data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1) # (B, T)
        data[1][1] = np.expand_dims(data[1][1], axis=-1) # (B, T, 1)
        return data

    train_raw = extend_labels(train_raw)
    val_raw = extend_labels(val_raw)

if args.mode == 'train':

    # Prepare training
    path = 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state'

    metrics_callback = keras_utils.ReadmissionMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose)
    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    if not os.path.exists('keras_logs'):
        os.makedirs('keras_logs')
    csv_logger = CSVLogger(os.path.join('keras_logs', model.final_name + '.csv'),
                           append=True, separator=';')

    print ("==> training")
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              nb_epoch=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)

elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw


    test_reader = ReadmissionReader(dataset_dir='/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/data/',
                                    listfile='/Users/jeffrey0925/MIMIC-III-clean/readmission_cv2/0_test_listfile801010.csv')

    ret = utils.load_data(test_reader, discretizer, normalizer,return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join("test_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")
