import argparse

from PIL import Image
import numpy as np

from cmz_comp.nn import NN
from cmz_comp.utils import (
    MAGIC_VALUE, PACK_QUANT, PACK_SHAPE, PACK_FLOAT, PACK_BOOL,
    PACK_FLOAT_SIZE, PACK_SHAPE_SIZE, PACK_BOOL_SIZE, PACK_QUANT_SIZE,
    Patcher, measured_task,
)


def process_cmdline():
    parser = argparse.ArgumentParser(description='Decompress grayscale image.')
    parser.add_argument('input_file_path', help='path to input file')
    parser.add_argument('output_file_path', help='path to output image')
    parser.add_argument(
        '--nn-decoder-activation', default='sigmoid',
        help='activation function of decoder network'
    )
    cargs = parser.parse_args()
    cargs = dict(vars(cargs))
    return cargs


class Decompressor:

    def __init__(self, input_path, output_path,
                 nn_decoder_activation='sigmoid'):
        self.input_path = input_path
        self.output_path = output_path
        self.nn_decoder_activation = nn_decoder_activation
        self.nn = None
        self.patcher = None
        self.input = None
        self.output = None
        self.img = None
        self.decode_matrices = None
        self.quantization_bins = None
        self.patch_x_size = None
        self.patch_y_size = None
        self.enc_patch_size = None
        self.dequantized_patches = None
        self.output_patches = None

    def prepare(self):
        self.input = open(self.input_path, 'rb')
        self.output = open(self.output_path, 'wb')

    def build_ae_patcher(self):
        self.nn = NN(
            patch_x_size=self.patch_x_size, patch_y_size=self.patch_y_size,
            enc_patch_size=self.enc_patch_size
        )
        self.patcher = Patcher(
            img_x_size=self.img_x_size, img_y_size=self.img_y_size, img=None,
            patch_x_size=self.patch_x_size, patch_y_size=self.patch_y_size
        )

    def read_input_float(self):
        return PACK_FLOAT.unpack(self.input.read(PACK_FLOAT_SIZE))[0]

    def read_input_shape(self):
        return PACK_SHAPE.unpack(self.input.read(PACK_SHAPE_SIZE))[0]

    def read_input_bool(self):
        return PACK_BOOL.unpack(self.input.read(PACK_BOOL_SIZE))[0]

    def read_input_quant(self):
        return PACK_QUANT.unpack(self.input.read(PACK_QUANT_SIZE))[0]

    def dequantize_patch(self, patch):
        return np.array([self.quantization_bins[k - 1] for k in patch])

    def read_input(self):
        # read magic value
        magic_value = [
            ord(self.input.read(PACK_QUANT_SIZE)),
            ord(self.input.read(PACK_QUANT_SIZE)),
            ord(self.input.read(PACK_QUANT_SIZE))
        ]
        if magic_value != MAGIC_VALUE:
            raise Exception('File format not recognized! (%s)' % magic_value)
        # read decode matrices
        # read size of weight matrix
        weight_matrix_x_size = self.read_input_shape()
        weight_matrix_y_size = self.read_input_shape()
        # read and reconstruct weight matrix
        weight_matrix_input = []
        for _ in range(weight_matrix_x_size * weight_matrix_y_size):
            weight_matrix_input.append(self.read_input_float())
        weight_matrix = np.array(weight_matrix_input).reshape(
            weight_matrix_x_size, weight_matrix_y_size
        )
        # read size of bias vector
        bias_vector_size = self.read_input_shape()
        # read and reconstruct weight matrix
        bias_vector_input = []
        for _ in range(bias_vector_size):
            bias_vector_input.append(self.read_input_float())
        bias_vector = np.array(bias_vector_input)
        self.decode_matrices = [weight_matrix, bias_vector]
        # read and reconstruct quantization bins
        quantization_bins_size = self.read_input_shape()
        quantization_bins_input = []
        for _ in range(quantization_bins_size):
            quantization_bins_input.append(self.read_input_float())
        self.quantization_bins = np.round(
            np.array(quantization_bins_input), decimals=4
        )
        # write original image size
        self.img_x_size = self.read_input_shape()
        self.img_y_size = self.read_input_shape()
        # read and reconstruct original patch size
        self.patch_x_size = self.read_input_shape()
        self.patch_y_size = self.read_input_shape()
        # read and reconstruct encoded patch length
        self.enc_patch_size = self.read_input_shape()
        # read and reconstruct number of quantized patches
        encoded_patches_len = self.read_input_shape()
        # read and reconstruct quantized patches
        self.dequantized_patches = []
        for _ in range(encoded_patches_len):
            encoded_patch_input = []
            for __ in range(self.enc_patch_size):
                encoded_patch_input.append(ord(self.read_input_quant()))
            encoded_patch = np.array(encoded_patch_input)
            dequantized_patch = self.dequantize_patch(encoded_patch)
            self.dequantized_patches.append(dequantized_patch)
        self.dequantized_patches = np.vstack(self.dequantized_patches)

    def prepare_output(self):
        self.img = Image.new('L', (self.img_x_size, self.img_y_size), color=0)

    def decode(self):
        self.nn.decode(self.dequantized_patches)
        self.output_patches = []
        for output_patch_content in self.nn.decoded_patches:
            # restore proper pixel values
            output_patch_content = (
                np.ceil(output_patch_content * 255.).astype(int)
            )
            self.output_patches.append(output_patch_content)

    def clear_image(self, img, color=0):
        img.paste(color, [0, 0, img.size[0], img.size[1]])

    def write_output(self):
        # get original patch coordinates
        patch_coords = self.patcher.get_patch_coordinates()
        # buffer image
        pimg = Image.new('L', (self.patch_x_size, self.patch_y_size), color=0)
        for patch_coord, output_patch in zip(
                patch_coords, self.output_patches):
            self.clear_image(pimg)
            pimg.putdata(output_patch)
            self.img.paste(pimg, patch_coord)
        self.img.save(self.output)

    def finish(self):
        self.input.close()
        self.output.close()

    def run(self):
        self.prepare()
        measured_task(
            'Reading input... ', self.read_input
        )
        self.prepare_output()
        self.build_ae_patcher()
        measured_task(
            'Building decoder... ', self.nn.build_decoder,
            self.decode_matrices,
            nn_activation=self.nn_decoder_activation
        )
        measured_task(
            'Decoding image... ', self.decode,
        )
        measured_task(
            'Writing output... ', self.write_output
        )
        self.finish()


if __name__ == '__main__':
    cargs = process_cmdline()
    decompressor = Decompressor(**cargs)
    decompressor.run()
