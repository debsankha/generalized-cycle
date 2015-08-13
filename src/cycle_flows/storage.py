#!/usr/bin/env python
"""
    storage.py
    Contains functions pertaining to saving/loading data files.
"""

import os
import gzip
import bz2
import argparse

try:
    import cPickle as pickle
except:
    import pickle

def save(obj, fname):
    """
        Saves the given object to the given file
        using pickle and gzip (if the file name
        in .gz)
    """
    ext = os.path.splitext(fname)[1]
    if ext == '.gz':
        file = gzip.GzipFile(fname, 'wb')
    elif ext == '.bz2':
        file = bz2.BZ2File(fname, 'wb')
    else:
        file = open(fname, 'wb')
    
    # Use highest pickle protocol
    pickle.dump(obj, file, -1)
    file.close()

def load(fname):
    """
        Loads the object pickled in fname.
        Decompresses a gzipped file if it
        ends in .gz
    """
    ext = os.path.splitext(fname)[1]
    if ext == '.gz':
        file = gzip.GzipFile(fname, 'rb')
    elif ext == '.bz2':
        file = bz2.BZ2File(fname, 'rb')
    else:
        file = open(fname, 'rb')
    
    obj = pickle.load(file)

    file.close()

    if isinstance(obj, list) and len(obj) == 1:
        obj = obj[0]

    return obj
