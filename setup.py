#!/usr/bin/env python3
from os.path import join, basename, abspath, dirname
from setuptools import setup

with open(join(dirname(abspath(__file__)), 'requirements.txt')) as f:
    requirements = f.readlines()

setup(
    name='padatious',
    version='0.4.3',  # Also change in padatious/__init__.py
    description='A neural network intent parser',
    url='http://github.com/MycroftAI/padatious',
    author='Matthew Scholefield',
    author_email='matthew331199@gmail.com',
    license='Apache-2.0',
    packages=[
        'padatious'
    ],
    install_requires=requirements,
    zip_safe=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='intent-parser parser text text-processing',
)
