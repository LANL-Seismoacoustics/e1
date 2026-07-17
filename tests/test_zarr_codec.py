"""
Tests for e1 Zarr v3 codec.

These tests require zarr to be installed:
    pip install e1[zarr]
    
or:
    pip install zarr>=3.0.0
    
Note: Set async.concurrency=1 before running tests to ensure thread safety.
"""
import warnings

import numpy as np
import pytest

# Check if zarr is available
try:
    import zarr
    from zarr.abc.codec import ArrayBytesCodec
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

# Skip all tests if zarr not installed
pytestmark = pytest.mark.skipif(
    not ZARR_AVAILABLE,
    reason="zarr not installed (install with: pip install e1[zarr])"
)

import e1

if ZARR_AVAILABLE:
    try:
        from e1_zarr_codec import E1Codec
    except ImportError:
        E1Codec = None


# =============================================================================
# Test Data
# =============================================================================

# Random data generator
RNG = np.random.default_rng(42)


def rand_int32(size):
    """Generate random int32 data."""
    if isinstance(size, tuple):
        total = np.prod(size)
        return RNG.integers(-1000000, 1000000, total, dtype=np.int32).reshape(size)
    else:
        return RNG.integers(-1000000, 1000000, size, dtype=np.int32)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="module", autouse=True)
def set_safe_concurrency():
    """Set Zarr concurrency to 1 for all tests to ensure thread safety."""
    if ZARR_AVAILABLE:
        original = zarr.config.get('async.concurrency')
        zarr.config.set({'async.concurrency': 1})
        yield
        # Reset to original after tests
        zarr.config.set({'async.concurrency': original})
    else:
        yield


# =============================================================================
# Codec Instantiation and Configuration Tests
# =============================================================================

def test_codec_import():
    """Test that E1Codec can be imported."""
    assert E1Codec is not None, "E1Codec should be importable"


def test_codec_instantiation():
    """Test creating an E1Codec instance."""
    codec = E1Codec()
    assert codec is not None
    assert isinstance(codec, ArrayBytesCodec)


def test_codec_id():
    """Test that codec has correct ID."""
    codec = E1Codec()
    assert codec.codec_id == "e1"


def test_get_config():
    """Test get_config returns empty dict (no parameters)."""
    codec = E1Codec()
    config = codec.get_config()
    assert isinstance(config, dict)
    assert len(config) == 0


def test_from_config():
    """Test creating codec from config."""
    codec = E1Codec.from_config({})
    assert codec is not None
    assert isinstance(codec, E1Codec)


def test_config_roundtrip():
    """Test get_config/from_config roundtrip."""
    codec1 = E1Codec()
    config = codec1.get_config()
    codec2 = E1Codec.from_config(config)
    assert codec2.get_config() == config


# =============================================================================
# Zarr Array Creation Tests
# =============================================================================

def test_create_zarr_array_with_e1_codec():
    """Test creating a Zarr array with E1Codec."""
    store = zarr.storage.MemoryStore()
    
    array = zarr.create(
        shape=(1000,),
        chunks=(100,),
        dtype='int32',
        store=store,
        codecs=[E1Codec()]
    )
    
    assert array is not None
    assert array.shape == (1000,)
    assert array.dtype == np.int32


def test_zarr_array_single_chunk():
    """Test creating Zarr array with single chunk (always safe)."""
    store = zarr.storage.MemoryStore()
    
    array = zarr.create(
        shape=(1000,),
        chunks=(1000,),  # Single chunk
        dtype='int32',
        store=store,
        codecs=[E1Codec()]
    )
    
    assert array.shape == (1000,)
    assert array.chunks == (1000,)


def test_wrong_dtype_raises_error():
    """Test that non-int32 dtype raises ValueError."""
    store = zarr.storage.MemoryStore()
    
    with pytest.raises(ValueError, match="only supports int32"):
        zarr.create(
            shape=(100,),
            chunks=(100,),
            dtype='float32',  # Wrong type
            store=store,
            codecs=[E1Codec()]
        )


# =============================================================================
# Data Read/Write Tests
# =============================================================================

def test_write_read_1d_array():
    """Test writing and reading 1D data through Zarr array with multiple chunks."""
    store = zarr.storage.MemoryStore()
    
    array = zarr.create(
        shape=(1000,),
        chunks=(100,),  # 10 chunks
        dtype='int32',
        store=store,
        codecs=[E1Codec()]
    )
    
    # Write data
    data = rand_int32(1000)
    array[:] = data
    
    # Read back
    retrieved = array[:]
    np.testing.assert_array_equal(retrieved, data)


def test_write_read_2d_array():
    """Test 2D Zarr array with E1Codec."""
    store = zarr.storage.MemoryStore()
    
    array = zarr.create(
        shape=(100, 50),
        chunks=(50, 25),
        dtype='int32',
        store=store,
        codecs=[E1Codec()]
    )
    
    # Write data
    data = rand_int32((100, 50))
    array[:, :] = data
    
    # Read back
    retrieved = array[:, :]
    np.testing.assert_array_equal(retrieved, data)


def test_write_read_3d_array():
    """Test 3D Zarr array with E1Codec."""
    store = zarr.storage.MemoryStore()
    
    array = zarr.create(
        shape=(20, 30, 40),
        chunks=(10, 15, 20),
        dtype='int32',
        store=store,
        codecs=[E1Codec()]
    )
    
    # Write data
    data = rand_int32((20, 30, 40))
    array[:, :, :] = data
    
    # Read back
    retrieved = array[:, :, :]
    np.testing.assert_array_equal(retrieved, data)


@pytest.mark.parametrize("size", [
    1,      # single value
    10,     # small
    100,    # medium
    510,    # exact block size
    511,    # one over block
    765,    # 1.5 blocks
    1020,   # 2 blocks
    2000,   # multiple blocks
])
def test_roundtrip_various_sizes(size):
    """Test encode/decode roundtrip with various array sizes via Zarr.
    
    Tests different data sizes to verify codec handles e1's internal block
    structure correctly (e1 uses 510-sample blocks). Single chunk per test
    to isolate size-related behavior from multi-chunk concurrency issues.
    """
    store = zarr.storage.MemoryStore()
    
    array = zarr.create(
        shape=(size,),
        chunks=(size,),  # Single chunk for simplicity
        dtype='int32',
        store=store,
        codecs=[E1Codec()]
    )
    
    # Generate random int32 data
    data = rand_int32(size)
    
    # Write and read
    array[:] = data
    retrieved = array[:]
    
    # Verify
    np.testing.assert_array_equal(retrieved, data)


# =============================================================================
# Configuration Warning Tests
# =============================================================================

def test_validation_warns_on_unsafe_config():
    """Test that validation warns when concurrency > 1 with multiple chunks."""
    # Temporarily set unsafe config
    original = zarr.config.get('async.concurrency')
    zarr.config.set({'async.concurrency': 10})
    
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            store = zarr.storage.MemoryStore()
            array = zarr.create(
                shape=(100, 100),
                chunks=(50, 50),  # Multiple chunks
                dtype='int32',
                store=store,
                codecs=[E1Codec()]
            )
            
            # Should have E1 warnings
            e1_warnings = [x for x in w if "E1 CODEC" in str(x.message)]
            assert len(e1_warnings) > 0, "Should warn about unsafe configuration"
            
            # Check warning content
            msg = str(e1_warnings[0].message)
            assert "async.concurrency: 10" in msg
            assert "config.set({'async.concurrency': 1})" in msg
    finally:
        # Reset to safe config
        zarr.config.set({'async.concurrency': original})


def test_validation_no_warning_on_safe_config():
    """Test that validation doesn't warn when concurrency = 1."""
    zarr.config.set({'async.concurrency': 1})
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        store = zarr.storage.MemoryStore()
        array = zarr.create(
            shape=(100, 100),
            chunks=(50, 50),  # Multiple chunks
            dtype='int32',
            store=store,
            codecs=[E1Codec()]
        )
        
        # Should not have E1 warnings
        e1_warnings = [x for x in w if "E1 CODEC" in str(x.message)]
        assert len(e1_warnings) == 0, "Should not warn with safe configuration"


def test_validation_no_warning_single_chunk():
    """Test that validation doesn't warn for single chunk arrays."""
    # Set unsafe concurrency
    original = zarr.config.get('async.concurrency')
    zarr.config.set({'async.concurrency': 10})
    
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            store = zarr.storage.MemoryStore()
            array = zarr.create(
                shape=(1000,),
                chunks=(1000,),  # Single chunk
                dtype='int32',
                store=store,
                codecs=[E1Codec()]
            )
            
            # Should not warn (single chunk is safe)
            e1_warnings = [x for x in w if "E1 CODEC" in str(x.message)]
            assert len(e1_warnings) == 0, "Should not warn with single chunk"
    finally:
        # Reset to safe config
        zarr.config.set({'async.concurrency': original})


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
