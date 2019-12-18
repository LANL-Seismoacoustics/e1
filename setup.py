#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
e1: Python support for the e1 compression format.


"""
import glob
from setuptools import setup, Extension

# from numpy.distutils.core import setup, Extension

with open('README.md') as readme:
    # https://dustingram.com/articles/2018/03/16/markdown-descriptions-on-pypi
    long_description = readme.read()

doclines = __doc__.split("\n")

setup(name='e1',
      version='0.1.0',
      description='Python support for the e1 compression format.',
      long_description=long_description,
      long_description_content_type="text/markdown", # setuptools >= 38.6.0
      author='Jonathan MacCarthy',
      author_email='jkmacc@lanl.gov',
      packages=['pisces', 'pisces.schema','pisces.io','pisces.tables',
                'pisces.commands'],
      url='https://github.com/LANL-seismoacoustics/e1',
      download_url='https://github.com/LANL-seismoacoustics/e1/tarball/0.1.0',
      keywords=['seismology', 'geophysics'],
      install_requires=['numpy'],
      ext_package='e1.lib',
      ext_modules=[Extension('libe1', ['src/e_compression.c'])],
      license='LANL-MIT',
      platforms=['Mac OS X', 'Linux/Unix'],
)