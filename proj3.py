# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 09:34:27 2019
@author: selle
"""
from  scipy import *
from  pylab import *
from mpi4py import MPI
from abc import ABC, abstractmethod
from room import LeftSideRoom, RightSideRoom, MiddleRoom
from scipy.linalg import toeplitz
from numpy.linalg import solve

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocessors = comm.Get_size()
print('helloworldfromprocess' , rank)

# Initialize the three rooms
LeftSideRoom = LeftSideRoom(height = 1, width = 2, deltaX = 1/2, gammaH = 40, gammaWF = 5, gammaNormal = 15)
RightSideRoom = RightSideRoom(height = 1, width = 1, deltaX = 1/3, gammaH = 40, gammaWF = 5, gammaNormal = 15)
MiddleRoom = MiddleRoom(height = 2, width = 1, deltaX = 1/5, gammaH = 40, gammaWF = 5, gammaNormal = 15)


class Solver:
    
    def __init__(self):
        pass
    
    def __call__(self, room, leftBoundaryCond = None, rightBoundaryCond = None):
        K = self.Kmatrix(room)
        F = self.Fvector(room, leftBoundaryCond, rightBoundaryCond)
        x = solve(K, F)
        return reshape(x, room.meshH-2, room.meshW-2)
    
    def initializeF(self, room):
        W = room.meshW-2
        H = room.meshH-2
        f = zeros(((room.meshW-2)*(room.meshH-2),1))
        roomType = type(room).__name__
        
        #room_reshape = append(room(),[])
        if roomType == 'MiddleRoom':
            upperCondition = -room.gammaH
            lowerCondition = -room.gammaWF
            BC_right = room.mesh[:,-1]
            BC_left = room.mesh[:,0]
            for i in range(room.meshW-2):
                f[i] += upperCondition
                f[-i-1] += lowerCondition
            for j in range(room.meshH-2):
                f[(room.meshW -3)+ j*(room.meshW-2)] += BC_right.T[i+1]
                f[-((room.meshW-2)*(j + 1))] += BC_left.T[-(i+2)]
           
        f = f/(room.deltaX**2)
        return f
    
    def Fvector(self, room, leftBoundaryCond, rightBoundaryCond):
        roomType = type(room).__name__
        F = self.initializeF(room)
        if roomType == 'LeftSideRoom':
            pass
        if roomType == 'RightSideRoom':
            pass
        if roomType == 'MiddleRoom':
            pass
        return F
    
    def Kmatrix(self, room):
        W = room.meshW - 2 # This is the width of the room without the borders
        H = room.meshH - 2
        K = toeplitz(concatenate([[-4,1],zeros(W-2),[1],zeros(W*H-W-1)]))

#        a = zeros((room.meshW-2)*(room.meshH-2))
#        a[0] = -4
#        a[1] = 1 
#        a[room.meshW-2] = 1
#        K = toeplitz(a)

        # This for-loop puts the zeros specific places in the matrix
        for i in range(W, W*H, W):
            K[i-1,i] -= 1
            K[i,i-1] -= 1
            
        return K # This is the Kmatrix without the borders

#a = zeros((5-2)*(5-2))
#a[0] = -4
#a[1] = 1 
#a[5-2] = 1
#k = toeplitz(a)
#        
solverObject = Solver()
K = solverObject.Kmatrix(MiddleRoom)
OldMesh = MiddleRoom()
NewMesh = solverObject(MiddleRoom)
