#!/usr/bin/env python

"""
    autosave.py

    provides a decorator which automatically
    save data points given to a matplotlib function
    as a text file. This way, plotted data is automatically
    and safely stored and can be replotted.
"""

import numpy as np

def asv(fname):
    def autosave_decorator(func):
        def wrapper(*args, **kwargs):
            # plot
            func(*args, **kwargs)
            
            # save
            if len(args) >= 2:
                # we have x and y coordinates, save them!
                x, y = args[:2]
                x = np.array(x)
                y = np.array(y)

                arr = np.vstack((x, y))
                np.savetxt(fname, arr.T)

        return wrapper
    return autosave_decorator
