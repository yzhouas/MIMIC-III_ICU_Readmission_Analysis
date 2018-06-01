from __future__ import absolute_import
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Convolution1D, MaxPooling1D, Flatten

from keras.engine import merge



class Network(Model):
    def __init__(self, dim, batch_norm, dropout, rec_dropout, task,
                 target_repl=False, deep_supervision=False, num_classes=1,
                 depth=1, input_dim=376, **kwargs):

        print("==> not used params in network class:", kwargs.keys())

        self.output_dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth
        self.dropout_words = 0.3
        self.dropout_rnn_U = 0.3
        self.drop_conv = 0.5

        if task in ['decomp', 'ihm', 'ph']:
            final_activation = 'sigmoid'
        elif task in ['los']:
            if num_classes == 1:
                final_activation = 'relu'
            else:
                final_activation = 'softmax'
        else:
            return ValueError("Wrong value for task")

        # Input layers and masking
        X = Input(shape=(48, input_dim), name='X')
        inputs = [X]

        nfilters = [2, 3, 4]
        nb_filters = 100
        pooling_reps = []
        for i in nfilters:
            feat_maps = Convolution1D(nb_filter=nb_filters,
                                      filter_length=i,
                                      border_mode="valid",
                                      activation="relu",
                                      subsample_length=1)(X)
            pool_vecs = MaxPooling1D(pool_length=2)(feat_maps)

            pool_vecs = Convolution1D(nb_filter=nb_filters,
                                      filter_length=i,
                                      border_mode="valid",
                                      activation="relu",
                                      subsample_length=1)(pool_vecs)
            pool_vecs = MaxPooling1D(pool_length=2)(pool_vecs)

            pool_vecs = Flatten()(pool_vecs)
            pooling_reps.append(pool_vecs)

        representation = merge(pooling_reps, mode='concat')

        representation = Dropout(self.drop_conv)(representation)

        y = Dense(num_classes, activation=final_activation)(representation)
        outputs = [y]

        return super(Network, self).__init__(inputs, outputs)



    def say_name(self):
        self.network_class_name = "k_lstm"
        return "{}.n{}{}{}{}.dep{}".format(self.network_class_name,
                                           self.output_dim,
                                           ".bn" if self.batch_norm else "",
                                           ".d{}".format(self.dropout) if self.dropout > 0 else "",
                                           ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                                           self.depth)


