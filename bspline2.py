# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:29:18 2019
@author: selle
"""
from  scipy import *
from  pylab import *
import matplotlib.pyplot as plt

class bspline:
    def __init__(self, u_grid, d):
        #self.coord = coord
        self.u_grid = u_grid
        self.d = d

    def __call__(self, u, d, blossom_i):
        self.u = u
        self.d = d
        i_hot = self.findhot(u)
        [d_hotx, d_hoty] = self.blossom(self.d, self.u, i_hot)
        sx = d_hotx[3, 0]
        sy = d_hoty[3, 0]
        s = (sx, sy)
        self.plot(blossom_i)
        print('s(', self.u, ') =', s)
        return s
    
    def returnd(self):
        """
        Returns d
        """
        return self.d
    
    def findhot(self, u):
        """
        Finds the hot interval for point u
        """
        i_hot = (self.u_grid > u).argmax() - 1
        return i_hot
    
    def blossom(self, d, u, i_hot):
        """
        Calculates the blossoms
        """
        d_hotx = zeros((4, 4))
        d_hoty = zeros((4, 4))
        d_x = [x[0] for x in d] 
        d_y = [x[1] for x in d] 
        hotstart = i_hot-2
        hotend = i_hot+2
        d_hotx[0,:] = d_x[hotstart:hotend]
        d_hoty[0,:] = d_y[hotstart:hotend]
        for t in range(0, 3): # (0, 1, 2)
            for index in range(0, 3-t): # (0, 1, 2) for t = 0; (0, 1) for t = 1; 0 for t = 2
                alpha = self.alpha(u, i_hot, t, index)
                d_hotx[t+1, index] = alpha*d_hotx[t, index] + (1-alpha)*d_hotx[t, index+1]
                d_hoty[t+1, index] = alpha*d_hoty[t, index] + (1-alpha)*d_hoty[t, index+1]
        return [d_hotx, d_hoty]
        
    def alpha(self, u, i_hot, t, index):
        """
        Calculates the alpha-values for blossom pairs
        """
        i_leftmost = i_hot - 2 + index + t
        u_leftmost = self.u_grid[i_leftmost]
        i_rightmost = i_hot +1 + index
        u_rightmost = self.u_grid[i_rightmost]
        return (u_rightmost - u) / (u_rightmost - u_leftmost)
    
    def plot(self, blossom_i, coord = 0):
        """
        Plots the spline, control points and blossoms
        """
        
        x_val = [x[0] for x in self.d]
        y_val = [x[1] for x in self.d]
        plt.plot(x_val,y_val,'or', label='Control points')
        u = linspace(0.0000001, 0.999999, 1000)
        s = zeros((1000, 2))
        for j in range(0, 1000):
            i_hot = self.findhot(u[j])
            [d_hotx, d_hoty] = self.blossom(self.d, u[j], i_hot)
            sx = d_hotx[3, 0]
            sy = d_hoty[3, 0]
            s[j, :] = (sx, sy)
        plt.plot(s[:,0], s[:,1], label='Spline')
        self.plotblossom(blossom_i)
        if coord != 0:
            x_coord = [x[0] for x in coord]
            y_coord = [x[1] for x in coord]
            plt.plot(x_coord,y_coord,'xb', label='Given coordinates')
            plt.suptitle('Spline created by interpolation')
        else:
            plt.suptitle('Spline created from given control points')
        plt.legend(loc = 'upper right')
        plt.show()
        
    def plotblossom(self, blossom_i): # Add-on 1
        """
        Plots the blossom curves
        """
        i_hot = self.findhot(self.u_grid[blossom_i])
        u = linspace(self.u_grid[i_hot]*0.95, self.u_grid[i_hot+1]*1.05, 100)
        d1 = zeros((100, 2))
        d2 = zeros((100, 2))
        d3 = zeros((100, 2))
        for j in range(0, 100):
            [d_hotx, d_hoty] = self.blossom(self.d, u[j], i_hot)
            d1x = d_hotx[1, 0]
            d1y = d_hoty[1, 0]
            d1[j, :] = (d1x, d1y)
            d2x = d_hotx[2, 0]
            d2y = d_hoty[2, 0]
            d2[j, :] = (d2x, d2y)
            d3x = d_hotx[3, 0]
            d3y = d_hoty[3, 0]
            d3[j, :] = (d3x, d3y)
        plt.plot(d1[:,0], d1[:,1], label='Linear blossom')
        plt.plot(d2[:,0], d2[:,1], label='Quadratic blossom')
        plt.plot(d3[:,0], d3[:,1], label='Final blossom')
        plt.legend(loc = 'upper right')
    
    def Nmatrix(self):
        N_matrix = zeros([len(self.u_grid)-2,len(self.u_grid)-2]) # Initializing the Nmatrix with zeros
        XI = self.ma_version1() # Here I'm creating the xi array
        
        for i in range(len(self.u_grid)-2):
            N_basis = self.basis(i+2)
            for j in range(len(self.u_grid)-2):
                N_matrix[j,i] = N_basis(XI[j])
        
        N_matrix[len(self.u_grid)-3,len(self.u_grid)-3] = 1 # Fuskar
        return N_matrix
    
    def ma_version1(self):
        """
        Moving average version 1
        """
        u_grid=self.u_grid
        xi=(u_grid[:-2] + u_grid[1:-1] + u_grid[2:])/3.
        return array(xi)
    
    def basis(self,j,k=3):
        """
        Runs the recursive algorithm for the basis functions. 
        This returns a function, not a number. On this function you can later 
        perform the N(u) method to get the function value.
        """
        u_grid = self.u_grid
        
        u_grid = append(u_grid,[u_grid[-1],u_grid[-1]])
        #u_grid = append(u_grid,[0,0])
        u_grid = insert(u_grid,0,[0,0])
    
        def N(u):
            """
            This is a nested method that performs the actual recursion and
            returns a scalar.
            """
            if k == 0:
                if u_grid[j-1] == u_grid[j]:
                    return 0.
                elif u >= u_grid[j-1] and u < u_grid[j]:
                    return 1.
                else:
                    return 0.
                
            if u_grid[j+k-1] == u_grid[j-1]:
                a = 0.
            else:
                a = (u-u_grid[j-1])/(u_grid[j+k-1]-u_grid[j-1])
            if (j+k)>len(u_grid)-1 or u_grid[j+k] == u_grid[j]:
                b = 0.
            else:
                b = (u_grid[j+k]-u)/(u_grid[j+k]-u_grid[j])  
  
            return a*self.basis(j, k-1)(u) + b*self.basis(j+1, k-1)(u)
        return N