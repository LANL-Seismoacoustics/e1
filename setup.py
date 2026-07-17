#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
e1: Python support for the e1 compression format.

This package provides Python support for the e1 seismic compression format,
including core compression/decompression functions and optional Zarr v3 codec.
"""
from setuptools import setup, Extension, find_packages

with open('README.md') as readme:
    # https://dustingram.com/articles/2018/03/16/markdown-descriptions-on-pypi
    long_description = readme.read()

doclines = __doc__.split("\n")

setup(name='e1',
      version='0.3.0',
      description='Python support for the e1 compression format.',
      long_description=long_description,
      long_description_content_type="text/markdown", # setuptools >= 38.6.0
      author='Jonathan MacCarthy',
      author_email='jkmacc@lanl.gov',
      url='https://github.com/LANL-seismoacoustics/e1',
      download_url='https://github.com/LANL-seismoacoustics/e1/tarball/0.3.0',
      keywords=['seismology', 'geophysics', 'compression', 'zarr'],
      install_requires=['numpy'],
      extras_require={
          'zarr': ['zarr>=3.0.0'],
      },
      packages=find_packages(),
      ext_modules=[Extension('_libe1', ['src/e_compression.c'])],
      entry_points={
          'zarr.codecs': [
              'e1=e1.codec:E1Codec',
          ],
      },
      license='MIT',
      platforms=['Mac OS X', 'Linux/Unix'],
)
