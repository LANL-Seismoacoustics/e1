"""
e1 : Python support for the e1 compression format

"""
import ctypes
from enum import IntEnum
import os
from math import ceil
import importlib.machinery

import numpy as np

ext = importlib.machinery.EXTENSION_SUFFIXES[0]
libecomp = ctypes.CDLL(os.path.dirname(__file__) + os.path.sep + '_libe1' + ext)
libecomp.e_decomp.restype = ctypes.c_int

libecomp.e_comp.argtypes = [
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_char*2,
    ctypes.c_int32
]
libecomp.e_comp.restype = ctypes.c_int
    # int32_t e_comp(int32_t *in,
    #                uint32_t *out,
    #                int32_t insamp,
    #                int32_t *outbytes,
    #                char datatype[],
    #                int32_t block_flag) {


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

E_MESSAGES = [
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

def decompress(buff, count):
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
    flen = len(buff) # number of bytes in buffer
    w = np.frombuffer(buff, dtype=np.int32) # read them all into 4byte integers
    # make an empty array to hold decompressed values
    Y = np.zeros(count, dtype=np.int32, order='C')
    # decompress into output array
    # int32_t e_decomp(uint32_t *in,
    #                  int32_t *out,
    #                  int32_t insamp,
    #                  int32_t inbyte, int32_t out0,
    #                  int32_t outsamp) {
    status = libecomp.e_decomp(w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                               Y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), count,
                               flen, 0, count)

    if status != ECStatus.EC_SUCCESS:
        msg = "e1 decompression error: {} {!r}".format(E_MESSAGES[status], ECStatus(status))
        raise Exception(msg)

    return Y


def compress(data):
    """ Compress an int32 array. 

    Parameters
    ----------
    data : numpy.ndarray (rank 1) of type int32

    Returns
    -------
    compressed : bytes
        Compressed data.

    """
    # out_c = ctypes.create_string_buffer(data.nbytes) # null-terminated mutable bytes buffer, max possible size (no compression)
    out = np.zeros(len(data), dtype='>u4', order='C') # big-endian unsigned int32 buffer array

    # int32_t e_comp(int32_t *in, uint32_t *out, int32_t insamp, int32_t *outbytes, char datatype[], int32_t block_flag)
    # int32_t e_comp(int32_t *in, 
    #                uint32_t *out, 
    #                int32_t insamp, 
    #                int32_t *outbytes, 
    #                char datatype[], 
    #                int32_t block_flag)

    # bytes_out_p = ctypes.POINTER(ctypes.c_uint32)() # null pointer to uint32
    outbytes_p = ctypes.POINTER(ctypes.c_int32)(ctypes.c_int32(0)) # initialize output bytes pointer
    datatype = ctypes.create_string_buffer(b'e1', 2)
    status = libecomp.e_comp(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        # bytes_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        # ctypes.cast(bytes_out, ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_int32(len(data)),
        outbytes_p,
        datatype,
        1
    )
    if status != ECStatus.EC_SUCCESS:
        msg = "e1 decompression error: {} {!r}".format(E_MESSAGES[status], ECStatus(status))
        raise Exception(msg)

    print(out)
    print(outbytes_p.contents.value)
    return out.tobytes()[:outbytes_p.contents.value]


def decompress_file(fobj, count):
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
