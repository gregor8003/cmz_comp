cmz_comp
========

CMZ (Cottrell, Munro, Zipser) is a trivial image compressor that uses
autoencoder neural network. Currently it handles geyscale PNG images.

Neural network is built in Keras with tensorflow backend.

Installation
------------

- Install latest `Anaconda3 <https://www.anaconda.com/download>`_

- Download and unpack latest cmz_comp source distribution, or simply clone the
  repo

- Create conda environment

.. code-block:: bash

    $ cd cmz_comp-<x.y.z.www>
    $ conda env create -f environment.yml
    $ activate cmz_comp

- Install cmz_comp

.. code-block:: bash

    $ python setup.py install

- Make sure that environment variable KERAS_BACKEND is set to "tensorflow"
  when environment is activated

Background
----------

The compressor extracts (X_src x Y_src) patches of pixels from source image,
and feeds them into autoencoder network that produces reduced representation of
patch. Typically, for 8x8 patches, reduced patch has 16 elements
(4x compression factor). Then, reduced patches are quantized and written to
output file, together with weight matrix of coding layer of autoencoder network.

The decompressor reads compressed file, reconstructs weight matrix, dequantizes
reduced patches, reconstructs (X_src x Y_src) patches of original size,
and writes them to output image.

Usage
-----

.. code-block:: bash

    $ python3 compressor.py -h

    usage: compressor.py [-h] [-px PATCH_X_SIZE] [-py PATCH_Y_SIZE]
                         [-encx ENC_PATCH_SIZE]
                         [--nn-ae-activation NN_AE_ACTIVATION]
                         [--nn-ae-optimizer NN_AE_OPTIMIZER]
                         [--nn-ae-loss NN_AE_LOSS] [--epochs NN_AE_TRAIN_EPOCHS]
                         [--batch-size NN_AE_TRAIN_BATCH_SIZE] [--noshuffle]
                         [--verbosity {0,1,2}]
                         [--nn-encoder-activation NN_ENCODER_ACTIVATION]
                         input_file_path output_file_path
    
    Compress grayscale image.
    
    positional arguments:
      input_file_path       path to input image
      output_file_path      path to output file
    
    optional arguments:
      -h, --help            show this help message and exit
      -px PATCH_X_SIZE, --patch-x-size PATCH_X_SIZE
                            width of image patch in pixels
      -py PATCH_Y_SIZE, --patch-y-size PATCH_Y_SIZE
                            height of image patch in pixels
      -encx ENC_PATCH_SIZE, --enc-patch-size ENC_PATCH_SIZE
                            length of encoded patch in pixels
      --nn-ae-activation NN_AE_ACTIVATION
                            activation function of autoencoder network
      --nn-ae-optimizer NN_AE_OPTIMIZER
                            optimizer function of autoencoder network
      --nn-ae-loss NN_AE_LOSS
                            loss function of autoencoder network
      --epochs NN_AE_TRAIN_EPOCHS, --nn-ae-train-epochs NN_AE_TRAIN_EPOCHS
                            number of training epochs for autoencoder network
      --batch-size NN_AE_TRAIN_BATCH_SIZE, --nn-ae-train-batch-size NN_AE_TRAIN_BATCH_SIZE
                            size of sample batch per weights update for training
                            of autoencoder network
      --noshuffle, --nn-ae-train-noshuffle
                            no sample shuffling for training of autoencoder
                            network
      --verbosity {0,1,2}, --nn-ae-train-verbosity {0,1,2}
                            verbosity of training of autoencoder network
      --nn-encoder-activation NN_ENCODER_ACTIVATION
                            activation function of encoder network

.. code-block:: bash

    $ python3 decompressor.py -h

    usage: decompressor.py [-h] [--nn-decoder-activation NN_DECODER_ACTIVATION]
                           input_file_path output_file_path
    
    Decompress grayscale image.
    
    positional arguments:
      input_file_path       path to input file
      output_file_path      path to output image
    
    optional arguments:
      -h, --help            show this help message and exit
      --nn-decoder-activation NN_DECODER_ACTIVATION
                            activation function of decoder network

References
----------

`https://blog.keras.io/building-autoencoders-in-keras.html <https://blog.keras.io/building-autoencoders-in-keras.html>`_

Cottrell G. W., Munro P., Zipser, D. "Image compression by back propagation:
An example of extensional programming", Models of cognition: rev. of
cognitive science, 1 (208), 1-4, 1989

Sample PNG files taken from: http://people.sc.fsu.edu/~jburkardt/data/png/png.html
