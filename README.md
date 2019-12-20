# e1.py

Python support for the e1 seismic compression format.


## Installation

```python
pip install e1
```

## Usage

```python
import e1

file_name = 'some_file.w'
byte_offset = 0
nsamples = 1000

data = e1.e_compression(file_name, byte_offset, nsamples)

```
