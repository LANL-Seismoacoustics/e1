# e1.py

Python support for the e1 seismic compression format.

"e1" is a variable-length compression algorithm for int32 data.


## Installation

### PyPI

```python
pip install e1
```

### Conda-Forge

**New**: now in Conda-Forge, thanks to @jcox10!

```python
conda install -c conda-forge e1
```

## Usage

### Decompress data from a file

```python
import e1

file_name = 'some_file.w'
byte_offset = 0
nsamples = 1000

with open(file_name, 'rb') as f:
    f.seek(byte_offset)
    data = e1.decompress_file(f, nsamples)

```

### Decompress raw bytes

```python
with open(file_name, 'rb') as f:
    # Read 5 times as many bytes as you expecte from nsamples x 4-byte values,
    # just to make sure all your nsamples are in it.  Though it may be more data
    # than you need, this gaurds against poorly-compressed data.  
    # In e1, you don't know a priori how many bytes it took to compress your data.
    nbytes = 5 * nsamples * 4
    byts = f.read(nbytes)

data = decompress(byts, nsamples)
```

### Compress a NumPy array

```python
import e1

e1bytes = e1.compress(my_array)

compression_ratio = my_array.nbytes / len(e1bytes)
print(f"Compression ratio: {compression_ratio}")

```

## Zarr Integration

The e1 codec is available as a Zarr v3 codec for seamless integration with Zarr arrays.

### Installation with Zarr support

```bash
pip install e1[zarr]
```

### Basic usage

```python
import zarr
import numpy as np
from zarr import config
from e1_zarr_codec import E1Codec

# IMPORTANT: Set concurrency to 1 for thread safety
config.set({'async.concurrency': 1})

# Create a Zarr array with e1 compression
array = zarr.create(
    shape=(1000, 500),
    chunks=(100, 50),
    dtype='int32',
    codecs=[E1Codec()]  # No BytesCodec needed - E1Codec is an ArrayBytesCodec
)

# Write data
data = np.random.randint(-1000, 1000, (1000, 500), dtype=np.int32)
array[:] = data

# Read data back
retrieved = array[:]
assert np.array_equal(data, retrieved)
```

### Thread Safety

**Important:** The e1 C library uses static buffers and is not thread-safe. When using with Zarr:

```python
from zarr import config
config.set({'async.concurrency': 1})  # Required for safe multi-chunk arrays
```

This serializes chunk operations to prevent data corruption. The codec will warn you if it detects an unsafe configuration.

See the [codec documentation](e1_zarr_codec.py) for detailed thread-safety information.
