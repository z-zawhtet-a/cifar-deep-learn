# This is simple code to test pushing code on Github
#
# Created by Titipat Achakulvisut, 29 July 2014

import numpy as np
import pandas as pd

m = 2
n = 2
sigma = 0.5

A = np.ones([m,n])
B = np.random.randn(n,m) + 1

C = np.dot(A, B)

print C
