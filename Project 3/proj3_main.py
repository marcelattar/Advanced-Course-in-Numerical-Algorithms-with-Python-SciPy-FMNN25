#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:12:04 2019

@author: Marcel
"""
from  scipy import *
from  pylab import *
from mpi4py import MPI
from abc import ABC, abstractmethod
from room import LeftSideRoom, RightSideRoom, MiddleRoom
from scipy.linalg import toeplitz
from numpy.linalg import solve
from proj3 import Solver
import matplotlib.pyplot as plt



"""
TYPE in terminal for program to run: 
    mpirun --oversubscribe -np 3 python /Users/Marcel/Documents/Python/NUMN25:FMNN25/numeriskaalgoritmer/'Project 3'/proj3_main.py
"""

#def main():

deltaX_in = 1/40
    
LeftSideRoom = LeftSideRoom(height = 1, width = 1, deltaX = deltaX_in, gammaH = 40, gammaWF = 5, gammaNormal = 15)
RightSideRoom = RightSideRoom(height = 1, width = 1, deltaX = deltaX_in, gammaH = 40, gammaWF = 5, gammaNormal = 15)
MiddleRoom = MiddleRoom(height = 2, width = 1, deltaX = deltaX_in, gammaH = 40, gammaWF = 5, gammaNormal = 15)

leftBC = 22*ones(int((MiddleRoom.meshH-3)/2))
rightBC = 23*ones(int((MiddleRoom.meshH-3)/2))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocessors = comm.Get_size()
testSolver = Solver()
#    print(rank)
#    print(nprocessors)
#    print('helloworldfromprocess' , rank)


for i in range(10):
    
    comm.barrier()

    if rank == 0:            
        leftBC = comm.recv(source=1)            
    
    elif rank == 1:
        MiddleRoom, leftBC, rightBC = testSolver(MiddleRoom,leftBC, rightBC)
        data_left = leftBC
        data_right = rightBC
        comm.send(data_left, dest=0)
        comm.send(data_right, dest=2)
        
    elif rank == 2:
        rightBC = comm.recv(source=1)

    
    
    
    if rank == 0:            
        LeftSideRoom, leftBC, rightBC = testSolver(LeftSideRoom, leftBoundaryCond = leftBC, rightBoundaryCond = None)
        data_left = leftBC
        comm.send(data_left, dest=1)
    
    elif rank == 1:       
        leftBC = comm.recv(source=0)
        rightBC = comm.recv(source=2)       
        
    elif rank == 2:
        RightSideRoom, leftBC, rightBC = testSolver(RightSideRoom, leftBoundaryCond = None, rightBoundaryCond = rightBC)
        data_right = rightBC
        comm.send(data_right, dest=1)
        
# Some kind of plot function now
if rank ==0:
    data_left = LeftSideRoom
    comm.send(data_left, dest=1)
    
elif rank ==1:
    leftRoom = comm.recv(source=0)
    rightRoom = comm.recv(source=2)
    
    allRooms = zeros([MiddleRoom.meshH, leftRoom.meshW*3])
    allRooms[leftRoom.meshH-1:, :leftRoom.meshW] = leftRoom.mesh
    allRooms[:, leftRoom.meshW:MiddleRoom.meshW*2] = MiddleRoom.mesh
    allRooms[:rightRoom.meshH, MiddleRoom.meshW*2:]  = rightRoom.mesh
    
    plt.imshow(allRooms, cmap='hot')
    plt.colorbar()
    plt.title('Temperatur in all rooms')
    plt.show()
    
elif rank ==2:
    data_right = RightSideRoom
    comm.send(data_right, dest=1)
    
#if _name_ == "_main_": 
 #   main()