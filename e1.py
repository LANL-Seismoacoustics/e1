import ctypes as C
import os
import sysconfig

import numpy as np

ext, = sysconfig.get_config_vars('SO')
libecomp = C.CDLL(os.path.dirname(__file__) + os.path.sep + '_libe1' + ext)

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


def e_compression(DATAFILE, BYTEOFFSET, NUM):
    """Wrapper to e1 decompression routine.

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

    # library call
    # replaces the first NUM values in w with decompressed values
    # retval = libecomp.e_decomp_inplace(w.ctypes.data_as(C.POINTER(C.c_int32)),
    #                                   NUM, flen, 0, NUM)
    retval = libecomp.e_decomp(w.ctypes.data_as(C.POINTER(C.c_uint32)),
                               Y.ctypes.data_as(C.POINTER(C.c_int32)), NUM,
                               flen, 0, NUM)

    # return w[0:NUM].copy(), retval
    # return w.copy(), retval
    return Y