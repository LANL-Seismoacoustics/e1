import io
import numpy as np
import e1

# e1 compressed data, 56 bytes (big endian).  wfid 200318583
e1bytes = (b'\x008\x00(\x01\x00\x01\x9c\x90\x10\x14Ax:$\xb0\xee\x97\x9a$\xc5\x1c'
           b'\xc2-\x05\xf5}T \x0bP\x00\xc9^V\xbd\xc6S:\xa0ww \xd1\xce\xbf\xa6R'
           b'\x81`\xf0M\xf0\x00\x00\x06')

# e1 compressed data, 56 bytes (big endian)
# wfid 200318583
e1bytes = (b'\x008\x00(\x01\x00\x01\x9c\x90\x10\x14Ax:$\xb0\xee\x97\x9a$\xc5\x1c'
           b'\xc2-\x05\xf5}T \x0bP\x00\xc9^V\xbd\xc6S:\xa0ww \xd1\xce\xbf\xa6R'
           b'\x81`\xf0M\xf0\x00\x00\x06')
# corresponding data for the 56 compressed e1 bytes above, 160 bytes uncompressed
e1data = np.array([257, 262, 327, 295, 248, 323, 352, 261, 210, 246, 286, 273,
                   277, 322, 345, 260, 217, 349, 351, 263, 263, 209, 202, 247,
                   308, 358, 306, 295, 327, 292, 221, 234, 291, 210, 165, 247,
                   269, 329, 406, 412], dtype=np.int32)

count = len(e1data)

e1file = io.BytesIO(e1bytes)

def test_decompress():
    observed = e1.decompress(e1bytes, count)
    np.testing.assert_array_equal(observed, e1data)


def test_decompress_file():
    observed = e1.decompress_file(e1file, count)
    np.testing.assert_array_equal(observed, e1data)

def test_compress():
    observed = e1.compress(e1data)
    assert observed == e1bytes


def test_compress_file():
    pass
