import argparse
import json
import os

from PIL import Image

from cmz_comp.nn import NN
from cmz_comp.utils import (
    MAGIC_VALUE, PACK_QUANT, PACK_SHAPE, PACK_FLOAT,
    QUANTIZATION_BINS_DEFAULT, Patcher, measured_task, plot_errors
)
import numpy as np


def process_cmdline():
    parser = argparse.ArgumentParser(description='Compress grayscale image.')
    parser.add_argument('input_file_path', help='path to input image')
    parser.add_argument('output_file_path', help='path to output file')
    parser.add_argument(
        '-px', '--patch-x-size', type=int, default=8, dest='patch_x_size',
        help='width of image patch in pixels'
    )
    parser.add_argument(
        '-py', '--patch-y-size', type=int, default=8, dest='patch_y_size',
        help='height of image patch in pixels'
    )
    parser.add_argument(
        '-encx', '--enc-patch-size', type=int, default=16,
        dest='enc_patch_size', help='length of encoded patch in pixels'
    )
    parser.add_argument(
        '--nn-ae-activation', default='sigmoid',
        help='activation function of autoencoder network'
    )
    parser.add_argument(
        '--nn-ae-optimizer', default='adadelta',
        help='optimizer function of autoencoder network'
    )
    parser.add_argument(
        '--nn-ae-loss', default='binary_crossentropy',
        help='loss function of autoencoder network'
    )
    parser.add_argument(
        '--epochs', '--nn-ae-train-epochs', type=int, default=50,
        dest='nn_ae_train_epochs',
        help='number of training epochs for autoencoder network'
    )
    parser.add_argument(
        '--batch-size', '--nn-ae-train-batch-size', type=int, default=1,
        dest='nn_ae_train_batch_size',
        help=(
            'size of sample batch per weights update '
            'for training of autoencoder network'
        )
    )
    parser.add_argument(
        '--noshuffle', '--nn-ae-train-noshuffle', action='store_true',
        dest='nn_ae_train_noshuffle',
        help=(
            'no sample shuffling for training of autoencoder network'
        )
    )
    parser.add_argument(
        '--verbosity', '--nn-ae-train-verbosity', type=int,
        choices=[0, 1, 2], default=0,
        dest='nn_ae_train_verbosity',
        help=(
            'verbosity of training of autoencoder network'
        )
    )
    parser.add_argument(
        '--nn-encoder-activation', default='sigmoid',
        help='activation function of encoder network'
    )
    parser.add_argument(
        '--dump-history-path', type=str, default='',
        dest='dump_history_path',
        help='dump training history to specified file'
    )
    parser.add_argument(
        '--dump-history-plots', action='store_true', default=False,
        dest='dump_history_plots',
        help='produce training history plots'
    )
    cargs = parser.parse_args()
    cargs = dict(vars(cargs))
    return cargs


class Compressor:

    def __init__(self, input_path, output_path='',
                 patch_x_size=8, patch_y_size=8, enc_patch_size=16,
                 quantization_bins=QUANTIZATION_BINS_DEFAULT,
                 nn_ae_activation='sigmoid', nn_ae_optimizer='adadelta',
                 nn_ae_loss='binary_crossentropy',
                 nn_ae_train_epochs=50, nn_ae_train_batch_size=1,
                 nn_ae_train_noshuffle=False, nn_ae_train_verbosity=0,
                 nn_encoder_activation='sigmoid',
                 dump_history_path='', dump_history_plots=False):
        self.input_path = input_path
        self.output_path = output_path
        self.patch_x_size = patch_x_size
        self.patch_y_size = patch_y_size
        self.enc_patch_size = enc_patch_size
        self.quantization_bins = quantization_bins
        self.nn_ae_activation = nn_ae_activation
        self.nn_ae_optimizer = nn_ae_optimizer
        self.nn_ae_loss = nn_ae_loss
        self.nn_ae_train_epochs = nn_ae_train_epochs
        self.nn_ae_train_batch_size = nn_ae_train_batch_size
        self.nn_ae_train_noshuffle = nn_ae_train_noshuffle
        self.nn_ae_train_verbosity = nn_ae_train_verbosity
        self.nn_encoder_activation = nn_encoder_activation
        self.dump_history_path = dump_history_path
        self.dump_history_plots = dump_history_plots
        self.input = None
        self.input_img = None
        self.output = None
        self.nn = None
        self.patcher = None

    def prepare(self):
        self.input = open(self.input_path, 'rb')
        if self.output_path != '':
            self.output = open(self.output_path, 'wb')
        self.input_img = Image.open(self.input)
        self.img_x_size, self.img_y_size = self.input_img.size
        self.build_ae_patcher()

    def build_ae_patcher(self):
        self.nn = NN(
            patch_x_size=self.patch_x_size, patch_y_size=self.patch_y_size,
            enc_patch_size=self.enc_patch_size
        )
        self.patcher = Patcher(
            img_x_size=self.img_x_size, img_y_size=self.img_y_size,
            img=self.input_img,
            patch_x_size=self.patch_x_size, patch_y_size=self.patch_y_size
        )

    def dump_history(self):
        self.dump_history_path = os.path.abspath(
            self.dump_history_path
        )
        os.makedirs(
            os.path.dirname(self.dump_history_path), exist_ok=True
        )
        with open(self.dump_history_path, 'w') as wf:
            json.dump(self.nn.history_ae(), wf, indent=2)

    def do_dump_history_plots(self):
        plot_output_path = '{}{}'.format(
            self.dump_history_path, '.plots.png'
        )
        plot_errors(self.nn.history_ae(), plot_output_path)

    def quantize_patch(self, comp_patch):
        return np.digitize(
            comp_patch, self.quantization_bins, right=False
        )

    def write_output(self):
        if self.output is not None:
            # write magic value
            for magic in MAGIC_VALUE:
                magic_byte = bytes([magic])
                self.output.write(PACK_QUANT.pack(magic_byte))
            # prepare decode matrices
            decode_matrices = self.nn.decode_matrices()
            # process weight matrix
            weight_matrix = decode_matrices[0]
            weight_matrix_x_size, weight_matrix_y_size = weight_matrix.shape
            flattened_weight_matrix = weight_matrix.ravel()
            # write size of weight matrix
            self.output.write(PACK_SHAPE.pack(weight_matrix_x_size))
            self.output.write(PACK_SHAPE.pack(weight_matrix_y_size))
            # write weight matrix
            for matrix_elem in flattened_weight_matrix:
                self.output.write(PACK_FLOAT.pack(matrix_elem))
            # process bias vector
            bias_vector = decode_matrices[1]
            bias_vector_size = len(bias_vector)
            # write length of bias vector
            self.output.write(PACK_SHAPE.pack(bias_vector_size))
            # write bias vector
            for vector_elem in bias_vector:
                self.output.write(PACK_FLOAT.pack(vector_elem))
            # write size of quantization bins table
            self.output.write(PACK_SHAPE.pack(len(self.quantization_bins)))
            # write quantization bins
            for qbin in self.quantization_bins:
                self.output.write(PACK_FLOAT.pack(qbin))
            # write original image size
            self.output.write(PACK_SHAPE.pack(self.img_x_size))
            self.output.write(PACK_SHAPE.pack(self.img_y_size))
            # write original patch size
            self.output.write(PACK_SHAPE.pack(self.patch_x_size))
            self.output.write(PACK_SHAPE.pack(self.patch_y_size))
            # write quantized patch length
            self.output.write(PACK_SHAPE.pack(self.enc_patch_size))
            # write number of quantized patches
            self.output.write(PACK_SHAPE.pack(len(self.nn.encoded_patches)))
            # quantize and write compressed patches
            for encoded_patch in self.nn.encoded_patches:
                quantized_patch = self.quantize_patch(encoded_patch)
                for quant in quantized_patch:
                    quant_bytes = bytes([quant])
                    self.output.write(PACK_QUANT.pack(quant_bytes))

    def finish(self):
        self.input.close()
        if self.output is not None:
            self.output.close()

    def run(self):
        self.prepare()
        measured_task(
            'Building network... ', self.nn.build_ae,
            nn_ae_activation=self.nn_ae_activation,
            nn_ae_optimizer=self.nn_ae_optimizer,
            nn_ae_loss=self.nn_ae_loss
        )
        measured_task(
            'Preprocessing image... ', self.patcher.build_patches_matrix
        )
        measured_task(
            'Training network... ', self.nn.train_ae,
            self.patcher.patches_matrix,
            epochs=self.nn_ae_train_epochs,
            batch_size=self.nn_ae_train_batch_size,
            shuffle=not self.nn_ae_train_noshuffle,
            verbosity=self.nn_ae_train_verbosity
        )
        if self.dump_history_path != '':
            measured_task(
                'Producing training history dump... ', self.dump_history,
            )
        if self.dump_history_plots:
            measured_task(
                'Producing training history plots... ',
                self.do_dump_history_plots,
            )
        measured_task(
            'Building encoder... ', self.nn.build_encoder,
            nn_activation=self.nn_encoder_activation
        )
        measured_task(
            'Encoding image... ', self.nn.encode,
            self.patcher.patches_matrix
        )
        measured_task(
            'Writing output... ', self.write_output
        )
        self.finish()


if __name__ == '__main__':
    cargs = process_cmdline()
    compressor = Compressor(**cargs)
    compressor.run()
