#!/usr/bin/env python

"""
Setup module for ModelAngelo
"""

import os
import sys

from setuptools import find_packages, setup

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
    packages=find_packages(),
    package_data={'': ['utils/stereo_chemical_props.txt']},
    version=model_angelo.__version__,
)
