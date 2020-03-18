"""
Author: Andrea Murphy
Date: Spring 2020
DESC: how to call a MATLAB script to compute the area of a triangle
from Python
"""

import matlab.engine
eng = matlab.engine.start_matlab()
eng.triarea(nargout=0)
eng.quit()
