#!/usr/bin/env python3
from os.path import join, basename, abspath, dirname
from setuptools import setup

with open(join(dirname(abspath(__file__)), 'requirements.txt')) as f:
    requirements = f.readlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='padatious',
    version='0.4.8',  # Also change in padatious/__init__.py
    description='A neural network intent parser',
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    entry_points={
        'console_scripts': [
            'padatious=padatious.__main__:main'
        ]
    },
    keywords='intent-parser parser text text-processing',
)
