#!/usr/bin/env python

"""
Setup module for ModelAngelo
"""

import os
import sys

from setuptools import setup, find_packages

sys.path.insert(0, f"{os.path.dirname(__file__)}/model_angelo")

import model_angelo

project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)), "model_angelo")

setup(
    name="model_angelo",
    entry_points={
        "console_scripts": [
            "model_angelo = model_angelo.__main__:main",
        ],
    },
    package_data={'': ['utils/stereo_chemical_props.txt']},
    packages=find_packages(),
    version=model_angelo.__version__,
    install_requires=[
        "tqdm",
        "scipy",
        "biopython>=1.81",
        "einops",
        "matplotlib",
        "mrcfile",
        "pandas",
        "fair-esm==1.0.3",
        "pyhmmer>=0.10.1",
        "loguru",
    ],
)
