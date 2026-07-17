"""e1 : Python support for the e1 compression format

This package provides Python support for the e1 seismic compression format,
including core compression/decompression functions and optional Zarr v3 codec.

Core compression/decompression
------------------------------
The main compression and decompression functions are available at the package level:

- compress: Compress int32 numpy arrays to e1 format
- decompress: Decompress e1 bytes to int32 numpy arrays
- decompress_file: Decompress from a file object
- e_compression: Legacy wrapper function

Exception Classes
-----------------
- E1Error: Base exception for e1 library errors
- E1CompressionError: Error during compression
- E1DecompressionError: Error during decompression
- E1ChecksumError: Checksum validation failed (data corrupted)
- E1ValidationError: Input validation failed

Constants
---------
- EC_FULL_END: Block end marker for full blocks
- EC_SHORT_END: Block end marker for short blocks
- BLOCK_SAMP: Samples per block (510)
- EC_MAX_BUFFER: Maximum samples (100000)
- ECStatus: Enum of status codes
- E_MESSAGES: Status code messages

Zarr Codec
----------
The E1Codec class is available for Zarr v3 integration:

- E1Codec: Zarr v3 ArrayBytesCodec for e1 compression

The codec can also be imported from the codec submodule:
    from e1.codec import E1Codec

Examples
--------
Basic compression and decompression:

>>> import numpy as np
>>> import e1
>>> 
>>> # Compress some data
>>> data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
>>> compressed = e1.compress(data)
>>> 
>>> # Decompress
>>> decompressed = e1.decompress(compressed, len(data))
>>> assert np.array_equal(data, decompressed)

Using with Zarr (requires zarr>=3.0.0):

>>> import zarr
>>> from zarr import config
>>> from e1 import E1Codec
>>> 
>>> # Set concurrency to 1 for thread safety
>>> config.set({'async.concurrency': 1})
>>> 
>>> # Create array with e1 compression
>>> array = zarr.create(
...     shape=(1000,),
...     chunks=(100,),
...     dtype='int32',
...     codecs=[E1Codec()]
... )
"""

# Import all public API from core module
from e1.core import (
    # Exceptions
    E1Error,
    E1CompressionError,
    E1DecompressionError,
    E1ChecksumError,
    E1ValidationError,
    
    # Constants
    EC_FULL_END,
    EC_SHORT_END,
    BLOCK_SAMP,
    EC_MAX_BUFFER,
    ECStatus,
    E_MESSAGES,
    
    # Functions
    compress,
    decompress,
    decompress_file,
    e_compression,
)

# Import codec for Zarr integration (optional dependency)
try:
    from e1.codec import E1Codec
    __all__ = [
        # Exceptions
        'E1Error',
        'E1CompressionError',
        'E1DecompressionError',
        'E1ChecksumError',
        'E1ValidationError',
        
        # Constants
        'EC_FULL_END',
        'EC_SHORT_END',
        'BLOCK_SAMP',
        'EC_MAX_BUFFER',
        'ECStatus',
        'E_MESSAGES',
        
        # Functions
        'compress',
        'decompress',
        'decompress_file',
        'e_compression',
        
        # Codec
        'E1Codec',
    ]
except ImportError:
    # zarr not installed, codec not available
    __all__ = [
        # Exceptions
        'E1Error',
        'E1CompressionError',
        'E1DecompressionError',
        'E1ChecksumError',
        'E1ValidationError',
        
        # Constants
        'EC_FULL_END',
        'EC_SHORT_END',
        'BLOCK_SAMP',
        'EC_MAX_BUFFER',
        'ECStatus',
        'E_MESSAGES',
        
        # Functions
        'compress',
        'decompress',
        'decompress_file',
        'e_compression',
    ]

# Package metadata
__version__ = '0.3.0'
