
# CHANGELOG

## 0.3.0

* **Package Structure**: Reorganized project into an `e1` package structure:
  - Moved `e1.py` → `e1/core.py` (internal)
  - Moved `e1_zarr_codec.py` → `e1/codec.py`
  - Created `e1/__init__.py` that exports all public APIs
  - All existing imports continue to work: `import e1`, `from e1 import compress, decompress`, etc.
  - `E1Codec` now imported as: `from e1 import E1Codec` (previously `from e1_zarr_codec import E1Codec`)
  - Updated setup.py to use `find_packages()` instead of `py_modules`
  - Updated all documentation, docstrings, and tests

## 0.3.0

* Added an optional Zarr compression codec, `E1Codec`.

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
