import os

from keras import metrics
from keras.layers import Dense
from keras.models import Sequential


# turn off tensorflow warnings about suboptimal architecture etc.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NN:

    def __init__(self, patch_x_size=8, patch_y_size=8, enc_patch_size=16):
        self.patch_x_size = patch_x_size
        self.patch_y_size = patch_y_size
        self.patch_size = self.patch_x_size * self.patch_y_size
        self.enc_patch_size = enc_patch_size
        self.ae_nn = None
        self.ae_nn_history = None
        self.encoder_nn = None
        self.decoder_nn = None
        self.encoded_patches = None
        self.decoded_patches = None

    def build_ae(self, nn_ae_activation='sigmoid', nn_ae_optimizer='adadelta',
                 nn_ae_loss='binary_crossentropy'):
        self.ae_nn = Sequential()
        self.ae_nn.add(
            Dense(
                self.enc_patch_size, activation=nn_ae_activation,
                input_shape=(self.patch_size,)
            )
        )
        self.ae_nn.add(
            Dense(
                self.patch_size, activation=nn_ae_activation
            )
        )
        self.ae_nn.compile(
            optimizer=nn_ae_optimizer, loss=nn_ae_loss,
            metrics=[metrics.MSE, metrics.MSLE, metrics.MAE, metrics.MAPE],
        )

    def train_ae(self, patches_matrix,
                 epochs=50, batch_size=1, shuffle=True, verbosity=0):
        self.ae_nn_history = self.ae_nn.fit(
            patches_matrix, patches_matrix,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            verbose=verbosity,  # 0 to turn off learning report
        )

    def history_ae(self):
        return self.ae_nn_history.history

    def encode_matrices(self):
        # return weight matrix and bias vector of first ae layer
        return self.ae_nn.layers[0].get_weights()

    def decode_matrices(self):
        # return weight matrix and bias vector of last ae layer
        return self.ae_nn.layers[-1].get_weights()

    def build_encoder(self, nn_activation='sigmoid'):
        self.encoder_nn = Sequential()
        self.encoder_nn.add(
            Dense(
                self.enc_patch_size, activation=nn_activation,
                input_shape=(self.patch_size,),
            )
        )
        self.encoder_nn.set_weights(self.encode_matrices())

    def encode(self, patches_matrix):
        self.encoded_patches = self.encoder_nn.predict(patches_matrix)

    def build_decoder(self, decode_matrices, nn_activation='sigmoid'):
        self.decoder_nn = Sequential()
        self.decoder_nn.add(
            Dense(
                self.patch_size, activation=nn_activation,
                input_shape=(self.enc_patch_size,),
            )
        )
        self.decoder_nn.set_weights(decode_matrices)

    def decode(self, compressed_patches_matrix):
        self.decoded_patches = (
            self.decoder_nn.predict(compressed_patches_matrix)
        )
