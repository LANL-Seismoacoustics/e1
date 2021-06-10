"""
e1 : Python support for the e1 compression format

"""
import ctypes as C
import os
import importlib.machinery

import numpy as np

ext = importlib.machinery.EXTENSION_SUFFIXES[0]
libecomp = C.CDLL(os.path.dirname(__file__) + os.path.sep + '_libe1' + ext)
libecomp.e_decomp.restype = C.c_int

STATUS_CODE = {
    0: 'EC_SUCCESS',
    1: 'EC_FAILED',
    2: 'EC_LENGTH_ERROR',
    3: 'EC_SAMP_ERROR',
    4: 'EC_DIFF_ERROR',
    5: 'EC_CHECK_ERROR',
    6: 'EC_ARG_ERROR',
    7: 'EC_TYPE_ERROR',
    8: 'EC_MEMORY_ERROR',
}


def decompress(buff, count):
    """
    Decompress count values from a bytes buffer

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
    status = libecomp.e_decomp(w.ctypes.data_as(C.POINTER(C.c_uint32)),
                               Y.ctypes.data_as(C.POINTER(C.c_int32)), count,
                               flen, 0, count)

    if status != 0:
        msg = "e1 decompression error: {}".format(STATUS_CODE[status])
        raise Exception(msg)

    return Y


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
    libecomp.e_decomp.restype = C.c_int

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

    retval = libecomp.e_decomp(w.ctypes.data_as(C.POINTER(C.c_uint32)),
                               Y.ctypes.data_as(C.POINTER(C.c_int32)), NUM,
                               flen, 0, NUM)

    return Y
