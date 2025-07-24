"""
e1 : Python support for the e1 compression format

"""
from typing import List, BinaryIO
import ctypes
from enum import IntEnum
import os
import importlib.machinery

import numpy as np


EC_FULL_END = 0
EC_SHORT_END = 1
BLOCK_SAMP = 510  # samples per 2048‑byte block for 'e1'

ext = importlib.machinery.EXTENSION_SUFFIXES[0]
libecomp = ctypes.CDLL(os.path.dirname(__file__) + os.path.sep + '_libe1' + ext)

class ECStatus(IntEnum):
    EC_SUCCESS = 0
    EC_FAILED = 1
    EC_LENGTH_ERROR = 2
    EC_SAMP_ERROR = 3
    EC_DIFF_ERROR = 4
    EC_CHECK_ERROR = 5
    EC_ARG_ERROR = 6
    EC_TYPE_ERROR = 7
    EC_MEMORY_ERROR = 8

E_MESSAGES: List[str] = [
    "operation succeeded",
    "operation failed",
    "number of bytes in data incorrect",
    "number of samples in data incorrect",
    "error in number of differences",
    "check value (last sample in block) incorrect",
    "error in arguments to function",
    "datatype incorrect",
    "memory allocation error",
    ]

libecomp.e_decomp.argtypes = [
    ctypes.POINTER(ctypes.c_uint32), # uint32_t *in
    ctypes.POINTER(ctypes.c_int32), # int32_t *out
    ctypes.c_int32, # int32_t insamp
    ctypes.c_int32, # int32_t inbyte
    ctypes.c_int32, # int32_t out0
    ctypes.c_int32, # int32_t outsamp
]
libecomp.e_decomp.restype = ctypes.c_int

def decompress(buff: bytes, count: int) -> np.ndarray:
    """ Decompress count values from a bytes buffer

    Parameters
    ----------
    buff : bytes
    count : int
        Number of expected values in the buffer.

    Returns
    -------
    data : numpy.ndarray (rank 1) of int32

    Raises
    ------
    Exception
        Error code from decompression library.

    """
    inbyte = len(buff) # number of bytes in buffer

    in_array = np.frombuffer(buff, dtype=np.int32) # read them all into 4byte integers
    in_ptr = in_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

    # make an empty array to hold decompressed values
    out_array = np.zeros(count, dtype=np.int32, order='C')
    out_ptr = out_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    # decompress into output array
    # int32_t e_decomp(uint32_t *in,
    #                  int32_t *out,
    #                  int32_t insamp,
    #                  int32_t inbyte, 
    #                  int32_t out0,
    #                  int32_t outsamp) {
    status = libecomp.e_decomp(in_ptr, out_ptr, count, inbyte, 0, count)

    if status != ECStatus.EC_SUCCESS:
        msg = "e1 decompression error: {} {!r}".format(E_MESSAGES[status], ECStatus(status))
        raise Exception(msg)

    return out_array


libecomp.e_comp.argtypes = [
    ctypes.POINTER(ctypes.c_int32), # int32_t *in
    ctypes.POINTER(ctypes.c_uint32), # uint32_t *out
    ctypes.c_int32, # int32_t insamp
    ctypes.POINTER(ctypes.c_int32), # int32_t *outbytes
    ctypes.c_char_p, # pointer to NUL‑terminated string
    ctypes.c_int32 # int32_t block_flag
]
libecomp.e_comp.restype = ctypes.c_int32  # int32_t


def compress(data: np.ndarray, datatype=b"e1"):
    parts = []
    i = 0
    insamp = len(data)

    while i + BLOCK_SAMP < insamp:  # all guaranteed‑full blocks
        parts.append(_compress_one_block(data[i:i+BLOCK_SAMP],
                                         datatype, block_flag=EC_FULL_END))
        i += BLOCK_SAMP

    # last (possibly short) block, set SHORT_END
    parts.append(_compress_one_block(data[i:], datatype,
                                     block_flag=EC_SHORT_END))
    return b"".join(parts)


def _compress_one_block(chunk, datatype, block_flag):
    out_bytes_est = 2048  # worst case
    out_buffer = np.zeros(out_bytes_est//4, dtype=np.uint32)
    out_bytes = ctypes.c_int32()
    status = libecomp.e_comp(
        chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_int32(len(chunk)),
        ctypes.byref(out_bytes),
        datatype,
        ctypes.c_int32(block_flag),
    )
    if status:
        raise RuntimeError(f"e1 compression error {status}")
    return out_buffer[: out_bytes.value//4].tobytes()


def decompress_file(fobj: BinaryIO, count: int) -> np.ndarray:
    foff = fobj.tell() # record the incoming byte offest
    flen = fobj.seek(0, os.SEEK_END) # get total file size
    fobj.seek(foff) # go back to the incoming offset
    flen -= foff # number of bytes left in file
    # read 5 times the number of expected samples, or the remaining bytes in file
    flen = 5 * count if flen > 5 * count else flen
    # read a conservatively-large buffer of flen 4-byte values
    byts = fobj.read(flen * 4)

    return decompress(byts, count)


def e_compression(DATAFILE, BYTEOFFSET, NUM):
    """Legacy wrapper to e1 decompression routine.
    
    Parameters
    ----------
    DATAFILE: string
        Full path to e1 file.
    BYTEOFFSET: int
        Number of bytes to start of the data.
    NUM: int
        Number of expected samples.

    Returns
    -------
    data: numpy.array (rank 1) of int32
        Uncompressed data vector.
    retval: int
        Return value/code. One of the following:

        * 0: 'EC_SUCCESS',
        * 1: 'EC_FAILED',
        * 2: 'EC_LENGTH_ERROR',
        * 3: 'EC_SAMP_ERROR',
        * 4: 'EC_DIFF_ERROR',
        * 5: 'EC_CHECK_ERROR',
        * 6: 'EC_ARG_ERROR',
        * 7: 'EC_TYPE_ERROR',
        * 8: 'EC_MEMORY_ERROR'

    Raises
    ------
    ValueError : BYTEOFFSET exceeds file size.

    Notes
    -----
    e1 decompression C library is written by Richard Stead, LANL.

    """
    libecomp.e_decomp.restype = ctypes.c_int

    # open file, query size, jump to offset
    f = open(DATAFILE, 'rb')
    flen = os.stat(DATAFILE).st_size
    if flen < BYTEOFFSET:
        raise ValueError("BYTEOFFSET exceeds file size.")
    f.seek(BYTEOFFSET)
    flen -= BYTEOFFSET
    # flen is capped at 5*NUM
    flen = 5 * NUM if flen > 5 * NUM else flen

    # Read the entire compressed chunk
    # w = (int32_t *)malloc((int)((flen + 3) / 4) * 4)
    # read(fd, w, flen)
    w = np.fromfile(f, count=flen, dtype=np.int32)  # XXX: check this
    f.close()

    Y = np.zeros(NUM, dtype=np.int32, order='C')

    retval = libecomp.e_decomp(w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                               Y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), NUM,
                               flen, 0, NUM)

    return Y
