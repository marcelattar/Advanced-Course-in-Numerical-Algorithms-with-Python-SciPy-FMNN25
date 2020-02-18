#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:38:02 2019

@author: Marcel
"""

#from mpi4py import MPI
#
#
#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#rank = comm.Get_rank()
#
#if rank == 0:
#   data = [(x+1)**x for x in range(size)]
#   print('we will be scattering:', data)
#else:
#   data = None
#   
#data = comm.scatter(data, root=0)
#print('rank',rank,'has data:',data)




from  scipy import *
from  pylab import *
from mpi4py import MPI
from abc import ABC, abstractmethod
from room import LeftSideRoom, RightSideRoom, MiddleRoom
from scipy.linalg import toeplitz
from numpy.linalg import solve
from proj3 import Solver
import matplotlib.pyplot as plt
import numpy as np


a = np.zeros([2,3])