# -*- coding: utf-8 -*-
# Copyright (C) 2024  Ather Abbas
from setuptools import setup

import os
fpath = os.path.join(os.getcwd(), "readme.md")
if os.path.exists(fpath):
    with open(fpath, "r") as fd:
        long_desc = fd.read()

setup(
    name='tabulight',

    version="0.1.0",

    description='tabulight: exploratory data analysis tool for tabular data',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/AtrCheema/tabulight',

    author='Ather Abbas',
    author_email='ather_abbas786@yahoo.com',

    classifiers=[
        "Development Status :: 5 - Production/Stable",

        'Natural Language :: English',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
 
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Information Analysis", 
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Utilities",
    ],

    packages=['tabulight'],

    install_requires=[
        'numpy<=2.0.1, >=1.17',
        'easy_mpl',
        'pandas<=2.1.4, >=1.0.0',
        'matplotlib<=3.9.0, >=3.4.0',
    ],
    extras_require={
        'all': ["numpy<=2.0.1, >=1.17",
                "scipy<=1.14, >=1.4",
                "easy_mpl",
                'pandas<=2.1.4, >=1.0.0',
                'matplotlib<=3.9.0, >=3.0.0',
                'seaborn<=1.0.0, >=0.9.0',
                ],
    }
)