
# CHANGELOG

## 0.2.0

* compression add by @samualchodur!

## 0.1.2

* Added a module definition to `e_compression.c`. This formally defines the lbrary as a Python module, fixing the `error LNK2001: unresolved external symbol PyInit__libe1` that occurs on Windows and allows it to compile correctly.
* Changed the method for determining compiled Python module file extensions in `setup.py` from `sysconfig.get_config_vars()` to `importlib.machinery`, which should fix importing problems on Python 3.6 and 3.7.

## 0.1.1

* Add tests
* Add `decompress` and `decompress_file` functions, which replace the
  deprecated `e_compression` function.

## 0.1.0

* Initial release.
