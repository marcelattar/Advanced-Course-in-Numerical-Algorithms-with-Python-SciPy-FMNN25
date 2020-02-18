# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 09:34:27 2019
@author: selle
"""

from  scipy import *
from  pylab import *
from mpi4py import MPI
from abc import ABC, abstractmethod

class Room(ABC):
    
    def __init__(self, height, width, deltaX, gammaH, gammaWF, gammaNormal):
        self.height = height
        self.width = width
        self.deltaX = deltaX
        self.gammaH = gammaH  # Condition for walls with heater
        self.gammaWF = gammaWF # Condition for wals with big window
        self.gammaNormal = gammaNormal # All walls except gammaH and gammaWF
        self.initialTemp = 20
        self.meshH = int(self.height/self.deltaX+1.) # Rows in mesh
        self.meshW = int(self.width/self.deltaX+1.) # Colons in mesh
        self.mesh = self.Discretize()
    
    def __call__(self):
        """
        Returns the mesh.
        """
        return self.mesh
    
    @abstractmethod
    def Discretize(self):
        """
        Creates a mesh. 
        """
        pass
    
    @abstractmethod
    def UpdateMesh(self):
        """
        Replaces the mesh with a new mesh.
        """
        pass
    
class LeftSideRoom(Room):
    
    def Discretize(self):
        if ((self.height/self.deltaX)%1 == 0) and ((self.width/self.deltaX)%1 == 0): # Checks that hight and width are divisible by deltaX
            mesh = self.initialTemp*ones((self.meshH, self.meshW)) # Creates a room with correct dimensions and constant temperature
            mesh[0,:] = self.gammaNormal # Sets boundary condition
            mesh[-1,:] = self.gammaNormal # Sets boundary condition
            mesh[:,0] = self.gammaH # Sets boundary condition
            return mesh
        else:
            print('Height or width is not evenly divisible by deltaX')
            
    def UpdateMesh(self, newMesh):
        self.mesh[1:-1,1:] = newMesh

class RightSideRoom(Room):
        

    def Discretize(self):
        if ((self.height/self.deltaX)%1 == 0) and ((self.width/self.deltaX)%1 == 0): # Checks that hight and width are divisible by deltaX
            mesh = self.initialTemp*ones((self.meshH, self.meshW)) # Creates a room with correct dimensions and constant temperature
            mesh[0,:] = self.gammaNormal # Sets boundary condition
            mesh[-1,:] = self.gammaNormal # Sets boundary condition
            mesh[:,-1] = self.gammaH # Sets boundary condition
            return mesh
        else:
            print('Height or width is not evenly divisible by deltaX')
            
    def UpdateMesh(self, newMesh):
        self.mesh[1:-1,:-1] = newMesh
   
class MiddleRoom(Room):

    def Discretize(self):
        if ((self.height/self.deltaX)%1 == 0) and ((self.width/self.deltaX)%1 == 0): # Checks that hight and width are divisible by deltaX
            mesh = self.initialTemp*ones((self.meshH, self.meshW)) # Creates a room with correct dimensions and constant temperature
            mesh[:,0] = self.gammaNormal # Sets boundary condition
            mesh[:,-1] = self.gammaNormal # Sets boundary condition
            mesh[-1,:] = self.gammaWF # Sets boundary condition
            mesh[0,:] = self.gammaH # Sets boundary condition
            return mesh
        else:
            print('Height or width is not evenly divisible by deltaX')
            
    def UpdateMesh(self, newMesh, leftBoundaryCondition, rightBoundaryCondition):
        self.mesh[1:-1,1:-1] = newMesh
        self.mesh[-len(leftBoundaryCondition):,0] = leftBoundaryCondition
        self.mesh[:len(rightBoundaryCondition),-1] = rightBoundaryCondition 