from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cmz_comp',
    version='1.0.0',
    description='Simple image compression by backpropagation',
    long_description=long_description,
    author='Grzegorz Zycinski',
    author_email='g.zycinski@gmail.com',
    url='http://github.com/gregor8003/cmz_comp/',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Archiving :: Compression',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='image compression neural networks',
    packages=find_packages(exclude=['cmz_comp.test']),
    scripts=[
        'cmz_comp/compressor.py',
        'cmz_comp/decompressor.py',
    ],
    zip_safe=False
)
