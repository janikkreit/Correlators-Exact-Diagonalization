from setuptools import setup, find_packages

setup(
    name='beehive',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21',
        'scipy',
        'matplotlib',
        'tqdm',
        'h5py',
    ],
)