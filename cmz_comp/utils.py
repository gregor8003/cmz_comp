import struct
import time
import math

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


MAGIC_VALUE = [ord('C'), ord('M'), ord('Z')]

QUANTIZATION_BINS_DEFAULT = np.round(np.linspace(0.0, 1.0, 256), decimals=4)

PACK_FLOAT = struct.Struct('=f')
PACK_FLOAT_SIZE = 4
PACK_SHAPE = struct.Struct('=i')
PACK_SHAPE_SIZE = 4
PACK_BOOL = struct.Struct('=?')
PACK_BOOL_SIZE = 1
PACK_QUANT = struct.Struct('=c')
PACK_QUANT_SIZE = 1


class Patcher:

    def __init__(self, img_x_size, img_y_size, img=None,
                 patch_x_size=8, patch_y_size=8):
        self.img_x_size = img_x_size
        self.img_y_size = img_y_size
        self.img = img
        self.patch_x_size = patch_x_size
        self.patch_y_size = patch_y_size

    def get_patch_coordinates(self):
        x_patches = int(self.img_x_size / self.patch_x_size)
        y_patches = int(self.img_y_size / self.patch_y_size)
        patch_coords = [
            (
                x * self.patch_x_size,
                y * self.patch_y_size,
                (x + 1) * self.patch_x_size,
                (y + 1) * self.patch_y_size
            )
            for y in range(y_patches)
            for x in range(x_patches)
        ]
        return patch_coords

    def build_patches_matrix(self):
        patch_coords = self.get_patch_coordinates()
        self.patches_matrix = np.array([])
        if self.img is not None:
            patch_vectors = []
            for patch_coord in patch_coords:
                # produce crops of original image
                patch_img = self.img.crop(patch_coord)
                # get raster of pixel data from patch image
                patch_vector = np.array(list(patch_img.getdata()))
                # normalize pixels (256 gray levels)
                patch_vector = patch_vector / 255.
                patch_vectors.append(patch_vector)
            self.patches_matrix = np.stack(patch_vectors, axis=0)


def measured_task(message, task_callable, *args, **kwargs):
    print(message, end='', flush=True)
    start = time.perf_counter()
    task_callable(*args, **kwargs)
    end = time.perf_counter()
    print('took', (end-start), 'sec(s)')


def plot_errors(history_dump, plot_output_path):

    # fit plot squarely with subplots (approx N x N) if possible
    # total number of plots
    nplots = len(history_dump)
    # number of columns
    cols = int(math.sqrt(nplots))
    # number of rows
    rows = int(math.ceil(nplots / cols))

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure()
    legend = [[], []]
    for i, (name, values) in enumerate(history_dump.items()):
        try:
            options = ERROR_PLOTS[name]
            ax = fig.add_subplot(gs[i])
            line, = ax.plot(
                range(1, len(values)+1), values,
                label=options['label'], color=options['color']
            )
            ax.set_xlabel('epoch')
            ax.set_ylabel(options['label'])
            legend[0].append(line)
            legend[1].append(options['legend_name'])
        except KeyError:
            pass
    lgd = fig.legend(
        legend[0], legend[1], loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        bbox_transform=fig.transFigure,
    )
    fig.tight_layout()
    # save figure and calculate size with legend outside subplots
    fig.savefig(
        plot_output_path, bbox_extra_artists=(lgd,), bbox_inches='tight'
    )


ERROR_PLOTS = {
    'loss': {
        'color': 'blue',
        'label': 'Loss',
        'legend_name': 'Loss',
    },
    'mean_absolute_error': {
        'color': 'red',
        'label': 'MAE',
        'legend_name': 'Mean Absolute Error',
    },
    'mean_absolute_percentage_error': {
        'color': 'green',
        'label': 'MAPE',
        'legend_name': 'Mean Absolute Percentage Error',
    },
    'mean_squared_error': {
        'color': 'cyan',
        'label': 'MSE',
        'legend_name': 'Mean Squared Error',
    },
    'mean_squared_logarithmic_error': {
        'color': 'magenta',
        'label': 'MSLE',
        'legend_name': 'Mean Squared Logarithmic Error',
    }
}
