import io
import pytest
import struct

import numpy as np

import e1

RNG2 = np.random.default_rng(20250716)


# e1 compressed data, 56 bytes (big endian)
# wfid 200318583
e1bytes = (b'\x008\x00(\x01\x00\x01\x9c\x90\x10\x14Ax:$\xb0\xee\x97\x9a$\xc5\x1c'
           b'\xc2-\x05\xf5}T \x0bP\x00\xc9^V\xbd\xc6S:\xa0ww \xd1\xce\xbf\xa6R'
           b'\x81`\xf0M\xf0\x00\x00\x06')
# zero bytes 4x larger than data, to test padding behavior
padding_bytes = bytes(len(e1bytes) * 4)


# corresponding data for the 56 compressed e1 bytes above, 160 bytes uncompressed
e1data = np.array([257, 262, 327, 295, 248, 323, 352, 261, 210, 246, 286, 273,
                   277, 322, 345, 260, 217, 349, 351, 263, 263, 209, 202, 247,
                   308, 358, 306, 295, 327, 292, 221, 234, 291, 210, 165, 247,
                   269, 329, 406, 412], dtype=np.int32)

count = len(e1data)


def rand(n):
    return RNG2.integers(-8_000_000, 8_000_000, n, dtype=np.int32)


CMP_DATASETS = {
    "< block": rand(20),
    "= block": rand(510),
    "tiny_tail": rand(515),
    "1.5 block": rand(765),
    "fixed": e1data
}


@pytest.fixture
def e1file():
    with io.BytesIO(e1bytes) as f:
        yield f


@pytest.fixture
def e1file_padded():
    with io.BytesIO(e1bytes + padding_bytes) as f:
        yield f


def test_decompress():
    observed = e1.decompress(e1bytes, count)
    np.testing.assert_array_equal(observed, e1data)

    # still works when extra bytes are present
    observed = e1.decompress(e1bytes + padding_bytes, count)
    np.testing.assert_array_equal(observed, e1data)


def test_decompress_file(e1file, e1file_padded):
    observed = e1.decompress_file(e1file, count)
    np.testing.assert_array_equal(observed, e1data)

    # still works when file contains extra bytes
    observed = e1.decompress_file(e1file_padded, count)
    np.testing.assert_array_equal(observed, e1data)


def _analyse_blocks(buf: bytes):
    """Return list [(hdr_len, used_nonzero, hdr_samp), ...] for every
    block.
    """
    out = []
    off = 0
    while off < len(buf):
        hdr_len, hdr_samp = struct.unpack_from(">HH", buf, off)
        blk_bytes = buf[off: off + hdr_len]
        used = 4 + len(blk_bytes[4:].rstrip(b"\x00"))
        out.append((hdr_len, used, hdr_samp))
        off += hdr_len
    return out


def _compress_roundtrip_case(arr, label):
    comp = e1.compress(arr)
    decomp = e1.decompress(comp, len(arr))
    np.testing.assert_array_equal(decomp, arr)

    blocks = _analyse_blocks(comp)
    print(f"\n{label:10}  total={len(comp)} B  blocks={len(blocks)}")
    for i, (hdr, used, samp) in enumerate(blocks):
        stat = "PASS" if hdr == used else "FAIL"
        print(f"  blk {i:<2} samp={samp:<4} hdr_len={hdr:<5} "
              f"used={used:<5} {stat}")


def test_compress_roundtrip():
    """Run round-trip test on 5 different array sizes.
    """
    for label, arr in CMP_DATASETS.items():
        _compress_roundtrip_case(arr, label)


if __name__ == "__main__":
    # TODO: Add decompress and other tests
    print("### Running compression tests... ###")
    test_compress_roundtrip()
    print("### ...All compression tests passed ###")
    print("All tests passed.")
