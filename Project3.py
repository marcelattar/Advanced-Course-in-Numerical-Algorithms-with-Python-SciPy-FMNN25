#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:37:01 2019

@author: Marcel
"""

from  scipy import *
#import numpy as np
from  pylab import *
from mpi4py import MPI
from abc import ABC, abstractmethod


#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#nprocessors = comm.Get_size()
#print('helloworldfromprocess' , rank)



class Room(ABC):
    
    @abstractmethod
    def __call__(self):
        """ 
        Return the matrix that represents the room temperatur.
        """
        pass
    
    @abstractmethod
    def Discretize(self):
        pass
    
    
class SideRoom(Room):
    
    def __init__(self, height, width, deltaX, gammaH, gammaWF, gammaNormal):
        self.height = height
        self.width = width
        self.deltaX = deltaX
        self.gammaH = gammaH
        self.gammaWF = gammaWF
        self.gammaNormal = gammaNormal
        self.mesh = self.Discretize()
        
        
    def __call__(self):
        return self.mesh
        
    def Discretize(self,hello):
        self.Matrix = 'dasnjdjsad'
        
    def UpdateMesh(self,newMesh):
        try:
            newMesh.shape == self.mesh.shape
            self.mesh = newMesh
        except:
            print('Dimensions  of the new mesh does not match old one.')
           
        
        
#TestRoom = SideRoom(10)
#TestRoom()

a = array([[1,2],[2,3],[2,3]])

print(1.%0.05)