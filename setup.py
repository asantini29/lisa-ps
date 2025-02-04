# -*- coding: utf-8 -*-

from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

def read_data(metadata):
    with open('lisaps/lisaps.py') as file:
        for line in file.readlines():
            if '__' + metadata + '__' in line and '__main__' not in line:
                return line.split('"')[1]

setup(
    name = read_data('name'),
    version = read_data('version'),
    license = read_data('license'),
    author = read_data('author'),
    author_email = read_data('author_email'),
    packages = [read_data('name')],
    description = read_data('description'),
    install_requires=requirements,
    zip_safe=False
)