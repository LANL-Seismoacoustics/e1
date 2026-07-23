"""e1.codec: Zarr v3 bytes-to-bytes codec for e1 compression.

This module provides a Zarr v3 codec for the e1 seismic compression format.
e1 is a variable-length compression algorithm for int32 data.

The E1Codec class can be imported from the top-level e1 package:
    from e1 import E1Codec

or from this submodule:
    from e1.codec import E1Codec
"""
import asyncio
import struct
from typing import Any
import warnings

import numpy as np
import zarr
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer, default_buffer_prototype

from e1 import core as e1
from e1.core import E1DecompressionError, E1ChecksumError, E1ValidationError


class E1Codec(ArrayBytesCodec):
    """
    Zarr v3 array-to-bytes codec for e1 compression.
    
    e1 is a variable-length compression algorithm for int32 seismic data.
    The C library handles endianness internally, producing big-endian
    compressed output while accepting and returning native byte order arrays.
    
    **CRITICAL THREAD-SAFETY WARNING**
    ----------------------------------
    The underlying e1 C library uses static buffers and is NOT thread-safe.
    Using this codec with Zarr's default concurrent processing (concurrency > 1)
    will cause data corruption due to race conditions.
    
    **Safe Configuration - Option 1: Disable Concurrency (Recommended)**
    
    Set Zarr's async concurrency to 1 to serialize all operations:
    
    >>> from zarr import config
    >>> config.set({'async.concurrency': 1})
    >>> 
    >>> # Now safe to use with any chunk configuration
    >>> array = zarr.create(
    ...     shape=(1000,),
    ...     chunks=(100,),  # Multiple chunks OK now
    ...     dtype='int32',
    ...     codecs=[E1Codec()]  # No BytesCodec needed!
    ... )
    
    This is the recommended approach as it:
    - Works with any chunk configuration
    - Prevents all race conditions
    - Is simple to implement
    - Only affects performance (serializes operations)
    
    **Safe Configuration - Option 2: Single Chunk Arrays**
    
    Use chunk size equal to array size (only for small arrays):
    
    >>> array = zarr.create(
    ...     shape=(1000,),
    ...     chunks=(1000,),  # chunk_shape == shape
    ...     dtype='int32',
    ...     codecs=[E1Codec()]  # No BytesCodec needed!
    ... )
    
    This works because there's nothing to process concurrently, but:
    - Not suitable for large arrays
    - Loses Zarr's chunking benefits
    - Must load entire array for any access
    
    **Unsafe Configuration (Default Zarr Settings)**
    
    >>> # DANGEROUS: Default concurrency=10 with multiple chunks
    >>> array = zarr.create(
    ...     shape=(1000,),
    ...     chunks=(100,),  # 10 chunks + concurrency=10 → corruption!
    ...     dtype='int32',
    ...     codecs=[E1Codec()]  # Will cause corruption!
    ... )
    >>> # This will cause random data corruption!
    
    **Checking Your Configuration**
    
    >>> from zarr import config
    >>> import numpy as np
    >>> 
    >>> # Check current concurrency setting
    >>> concurrency = config.get('async.concurrency')
    >>> print(f"Current concurrency: {concurrency}")
    >>> 
    >>> # Check if array has multiple chunks
    >>> shape = (1000, 500)
    >>> chunk_shape = (100, 50)
    >>> chunks_per_dim = tuple((s + c - 1) // c for s, c in zip(shape, chunk_shape))
    >>> total_chunks = np.prod(chunks_per_dim)
    >>> 
    >>> if total_chunks > 1 and concurrency > 1:
    ...     print(f"WARNING: {total_chunks} chunks with concurrency={concurrency}")
    ...     print("This configuration will cause data corruption!")
    ...     print("Solution: config.set({'async.concurrency': 1})")
    
    **What Happens with Concurrent Processing:**
    
    - Silent data corruption (random incorrect values)
    - Checksum errors: "check value incorrect <ECStatus.EC_CHECK_ERROR: 5>"
    - Non-reproducible failures (depends on thread timing)
    - Data loss in production systems
    
    **Root Cause:**
    
    From src/e_compression.c line 181:
    
        static int32_t unbuf[EC_MAX_BUFFER];  // Shared across all threads!
    
    When Zarr processes chunks concurrently (default behavior):
    
    1. Thread A starts decompressing chunk 0, writes to unbuf[]
    2. Thread B starts decompressing chunk 1, writes to SAME unbuf[]
    3. Thread A reads corrupted data (mix of chunks 0 and 1)
    4. Result: Random garbage data or checksum errors
    
    **Environment Variable Configuration:**
    
    You can also set concurrency via environment variable:
    
    .. code-block:: bash
    
        export ZARR_ASYNC_CONCURRENCY=1
        python your_script.py
    
    **Context Manager for Temporary Configuration:**
    
    >>> from zarr import config
    >>> 
    >>> # Temporarily disable concurrency for e1 operations
    >>> with config.set({'async.concurrency': 1}):
    ...     # Safe to use e1 codec here
    ...     array[:] = data
    ...     retrieved = array[:]
    >>> # Concurrency restored to previous value after exiting
    
    **Performance Implications:**
    
    Setting ``async.concurrency=1`` serializes all chunk operations:
    
    - **Pro**: Completely prevents race conditions
    - **Pro**: Simple to implement
    - **Con**: Slower than concurrent processing
    - **Con**: Cannot leverage multiple CPU cores
    
    For e1 specifically, the tradeoff is acceptable because:
    - E1 is designed for seismic data (typically sequential access patterns)
    - Compression/decompression is fast (bottleneck is usually I/O)
    - Data correctness is more important than speed
    
    **Alternative Solutions:**
    
    1. **Use thread-safe codecs** for multi-chunk arrays:
       - blosc, zstd, gzip are all thread-safe
       - Better suited for Zarr's concurrent processing
       - Recommended for large arrays with many chunks
    
    2. **Add thread locking to codec** (advanced):
       - Modify _encode_sync() and _decode_sync()
       - Use threading.Lock() around e1 library calls
       - Maintains concurrency for other operations
       - Requires codec modifications
    
    3. **Fix the C library** (long-term):
       - Remove static buffers
       - Use thread-local storage or dynamic allocation
       - Proper solution but requires C code changes
    
    Limitations
    -----------
    - Only supports int32 data
    - NOT thread-safe (static buffers in C library)
    - Requires ``config.set({'async.concurrency': 1})`` for safe multi-chunk use
    - Performance limited by serialization
    
    See Also
    --------
    zarr.config : Zarr configuration system
    THREAD_SAFETY_ANALYSIS.md : Detailed analysis of concurrency issues
    
    Examples
    --------
    **Example 1: Safe usage with concurrency disabled**
    
    >>> import zarr
    >>> import numpy as np
    >>> from zarr import config
    >>> from e1 import E1Codec
    >>> 
    >>> # Disable concurrency for thread safety
    >>> config.set({'async.concurrency': 1})
    >>> 
    >>> # Create array with multiple chunks (safe now)
    >>> store = zarr.storage.MemoryStore()
    >>> array = zarr.create(
    ...     shape=(1000,),
    ...     chunks=(100,),
    ...     dtype='int32',
    ...     store=store,
    ...     codecs=[E1Codec()]  # No BytesCodec needed!
    ... )
    >>> 
    >>> # Write and read data
    >>> data = np.arange(1000, dtype=np.int32)
    >>> array[:] = data
    >>> retrieved = array[:]
    >>> assert np.array_equal(data, retrieved)
    
    **Example 2: Using context manager**
    
    >>> from zarr import config
    >>> 
    >>> # Normal operations with default concurrency
    >>> other_array = zarr.open('data.zarr')  # Uses concurrency=10
    >>> 
    >>> # E1 operations with serialization
    >>> with config.set({'async.concurrency': 1}):
    ...     e1_array = zarr.open('seismic.zarr')  # Has E1Codec
    ...     data = e1_array[:]  # Safe, serialized
    >>> 
    >>> # Back to concurrent processing
    
    **Example 3: Checking current configuration**
    
    >>> from zarr import config
    >>> print(f"Concurrency: {config.get('async.concurrency')}")
    >>> # Concurrency: 10 (default, unsafe for e1)
    >>> 
    >>> config.set({'async.concurrency': 1})
    >>> print(f"Concurrency: {config.get('async.concurrency')}")
    >>> # Concurrency: 1 (safe for e1)
    """
    
    codec_id = "e1"
    
    def __init__(self) -> None:
        """
        Initialize E1 codec.
        
        The codec has no configuration parameters.
        """
        pass
    
    def get_config(self) -> dict[str, Any]:
        """
        Return codec configuration for serialization.
        
        Returns
        -------
        dict
            Empty dictionary (no configuration parameters)
        """
        return {}
    
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "E1Codec":
        """
        Create codec from configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary (ignored, no parameters)
            
        Returns
        -------
        E1Codec
            New codec instance
        """
        return cls()
    
    def validate(self, *, shape: tuple[int, ...], dtype, chunk_grid) -> None:
        """
        Validate that codec configuration is safe for the given array.
        
        Issues a warning if the array has multiple chunks and Zarr's async
        concurrency is greater than 1, as this can cause data corruption due
        to the non-thread-safe C library.
        
        Note: threading.max_workers does NOT provide sufficient protection.
        Only async.concurrency=1 guarantees thread safety.
        
        Parameters
        ----------
        shape : tuple[int, ...]
            Array shape
        dtype : np.dtype
            Array data type
        chunk_grid : ChunkGridMetadata
            Chunk grid metadata
        
        Raises
        ------
        ValueError
            If dtype is not int32
        
        Warnings
        --------
        UserWarning
            If array has multiple chunks and async.concurrency > 1
        """
        # Check dtype (Zarr v3 uses custom dtype classes)
        dtype_class_name = dtype.__class__.__name__
        if dtype_class_name != 'Int32':
            raise ValueError(
                f"E1Codec only supports int32 data, got {dtype_class_name}. "
                f"Please convert your data to int32 before using this codec."
            )
        
        # Calculate number of chunks
        chunk_shape = chunk_grid.chunk_shape
        chunks_per_dim = tuple(
            (s + c - 1) // c  # Ceiling division
            for s, c in zip(shape, chunk_shape)
        )
        total_chunks = int(np.prod(chunks_per_dim))
        
        # Get current concurrency setting from zarr.config
        concurrency = zarr.config.get('async.concurrency', 10)
        
        # Issue warning if unsafe configuration detected
        # Note: We only check async.concurrency because threading.max_workers
        # does NOT provide sufficient protection - the asyncio scheduler can
        # still cause race conditions even with a single worker thread.
        if total_chunks > 1 and concurrency > 1:
            warnings.warn(
                f"\n"
                f"{'='*70}\n"
                f"E1 CODEC THREAD-SAFETY WARNING\n"
                f"{'='*70}\n"
                f"Your configuration may cause DATA CORRUPTION:\n"
                f"  - Array shape: {shape}\n"
                f"  - Chunk shape: {chunk_shape}\n"
                f"  - Total chunks: {total_chunks}\n"
                f"  - Zarr async.concurrency: {concurrency}\n"
                f"\n"
                f"The e1 C library is NOT thread-safe. Multiple chunks with\n"
                f"concurrency > 1 can cause race conditions and corrupt data.\n"
                f"\n"
                f"REQUIRED SOLUTION:\n"
                f"  from zarr import config\n"
                f"  config.set({{'async.concurrency': 1}})\n"
                f"\n"
                f"This serializes all async operations, preventing corruption.\n"
                f"\n"
                f"Note: Setting threading.max_workers=1 is NOT sufficient!\n"
                f"Only async.concurrency=1 guarantees thread safety.\n"
                f"\n"
                f"ALTERNATIVE SOLUTIONS:\n"
                f"  1. Use single chunk: chunks={shape}\n"
                f"  2. Use thread-safe codec (blosc, zstd, gzip)\n"
                f"\n"
                f"See codec docstring for detailed information.\n"
                f"{'='*70}\n",
                UserWarning,
                stacklevel=4
            )
    
    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        """
        Compute encoded size for buffer allocation.
        
        e1 is a variable-length compression codec, so the output size cannot
        be determined without actually compressing the data.
        
        Parameters
        ----------
        input_byte_length : int
            Number of bytes in the input buffer
        chunk_spec : ArraySpec
            Chunk specification
            
        Raises
        ------
        NotImplementedError
            Always raised, as e1 produces variable-length output
        """
        raise NotImplementedError(
            "E1Codec produces variable-length output. "
            "The compressed size cannot be determined without encoding the data."
        )
    
    def _encode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        """
        Synchronous encoding (compression) implementation.
        
        Parameters
        ----------
        chunk_array : NDBuffer
            Input array-like buffer containing int32 data
        chunk_spec : ArraySpec
            Chunk specification
            
        Returns
        -------
        Buffer
            Compressed data with 8-byte header
            
        Raises
        ------
        E1ValidationError
            If data validation fails (from e1 library)
        E1CompressionError
            If e1 compression fails (from e1 library)
        """
        # Convert NDBuffer to numpy array and ensure C-contiguous in one step
        # This avoids double-copy: asarray with order='C' ensures contiguity
        np_array = np.asarray(chunk_array.as_numpy_array(), order='C')
        
        # Flatten multi-dimensional arrays (guaranteed to be a view since C-contiguous)
        data = np_array.ravel()
        
        # Compress using e1 (handles validation, endianness, and errors internally)
        # The C library produces big-endian compressed output
        compressed = e1.compress(data, datatype=b"e1")
        
        # Return as Buffer (no extra header - on-disk format matches original e1)
        return chunk_spec.prototype.buffer.from_bytes(compressed)
    
    async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        """
        Async wrapper for encoding a single chunk.
        
        Parameters
        ----------
        chunk_array : NDBuffer
            Input array-like buffer containing int32 data
        chunk_spec : ArraySpec
            Chunk specification
            
        Returns
        -------
        Buffer
            Compressed buffer
        """
        return await asyncio.to_thread(self._encode_sync, chunk_array, chunk_spec)
    
    def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> np.ndarray:
        """
        Synchronous decoding (decompression) implementation.
        
        Parameters
        ----------
        chunk_bytes : Buffer
            Compressed buffer with 8-byte header
        chunk_spec : ArraySpec
            Chunk specification
            
        Returns
        -------
        np.ndarray
            Decompressed int32 array
            
        Raises
        ------
        ValueError
            If input is too small, or if decompressed size doesn't match expected chunk size
        E1ChecksumError
            If decompression checksum validation fails (from e1 library)
        E1DecompressionError
            If e1 decompression fails for other reasons (from e1 library)
        E1ValidationError
            If input validation fails (from e1 library)
        """
        # Convert buffer to bytes (this is the raw e1 compressed byte stream)
        input_bytes = chunk_bytes.to_bytes()
        
        # Calculate expected sample count from chunk shape
        expected_size = int(np.prod(chunk_spec.shape))
        
        # Decompress using e1 (handles validation and typed errors)
        # e1.decompress expects the raw compressed bytes and the total sample count
        decompressed = e1.decompress(input_bytes, expected_size)
        
        # The decompressed array should be 1D with length equal to expected_size
        # Reshape to chunk shape and return
        return decompressed.reshape(chunk_spec.shape)
    
    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> np.ndarray:
        """
        Async wrapper for decoding a single chunk.
        
        Parameters
        ----------
        chunk_bytes : Buffer
            Compressed buffer
        chunk_spec : ArraySpec
            Chunk specification
            
        Returns
        -------
        np.ndarray
            Decompressed array
        """
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)
