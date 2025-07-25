#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e1: Python support for the e1 compression format.


"""
from setuptools import setup, Extension

# from numpy.distutils.core import setup, Extension

with open('README.md') as readme:
    # https://dustingram.com/articles/2018/03/16/markdown-descriptions-on-pypi
    long_description = readme.read()

doclines = __doc__.split("\n")

setup(name='e1',
      version='0.2.1',
      description='Python support for the e1 compression format.',
      long_description=long_description,
      long_description_content_type="text/markdown", # setuptools >= 38.6.0
      author='Jonathan MacCarthy',
      author_email='jkmacc@lanl.gov',
      url='https://github.com/LANL-seismoacoustics/e1',
      download_url='https://github.com/LANL-seismoacoustics/e1/tarball/0.1.0',
      keywords=['seismology', 'geophysics'],
      install_requires=['numpy'],
      py_modules=['e1'],
      ext_modules=[Extension('_libe1', ['src/e_compression.c'])],
      license='MIT',
      platforms=['Mac OS X', 'Linux/Unix'],
)
