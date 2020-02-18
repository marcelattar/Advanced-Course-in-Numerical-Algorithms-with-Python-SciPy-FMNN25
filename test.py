#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:21:12 2019
@author: mariagunnarsson
"""

import unittest
from bspline import *

class  TestBasis(unittest.TestCase):
    
        
    def setUp(self):
        self.d = [(-12.73564, 9.03455),
         (-26.77725, 15.89208),
         (-42.12487, 20.57261),
         (-15.34799, 4.57169),
         (-31.72987, 6.85753),
         (-49.14568, 6.85754),
        (-38.09753, -1e-05),
        (-67.92234, -11.10268),
        (-89.47453, -33.30804),
        (-21.44344, -22.31416),
        (-32.16513, -53.33632),
        (-32.16511, -93.06657),
        (-2e-05, -39.83887),
        (10.72167, -70.86103),
        (32.16511, -93.06658),
        (21.55219, -22.31397),
        (51.377, -33.47106),
        (89.47453, -33.47131),
        (15.89191, 0.00025),
        (30.9676, 1.95954),
        (45.22709, 5.87789),
        (14.36797, 3.91883),
        (27.59321, 9.68786),
        (39.67575, 17.30712)]
    
        u_grid = linspace(0, 1, 26)
        u_grid[ 1] = u_grid[ 2] = u_grid[ 0]
        u_grid[-3] = u_grid[-2] = u_grid[-1]
        self.spline = bspline(u_grid, self.d)       
        
    def testPositive(self):        
        d = self.spline.returnd()
        for j in range(len(self.d)):
            N = self.spline.basis(j)
            for i in self.spline.u_grid:
                Ni = N(i)
                self.assertFalse( Ni < 0 )
                
    def testdeBoor(self):
        s = self.spline(0.2, self.d, 15)
        self.assertEqual(s, (-31.902191666666667, 6.476558333333334))
        
    def testSum(self):
        N_sum = zeros(1000)
        u = linspace(0.00001, 0.999999, 1000)
        for j in range(len(self.spline.u_grid)):
            N = self.spline.basis(j)
            Ny = zeros(1000)
            for i in range(1000):
                Ny[i] = N(u[i])
            plt.plot(u, Ny)
            N_sum = N_sum + Ny
        plt.plot(u, N_sum)
        plt.show()
        self.assertEqual(N_sum.all(), 1)
    
    def testNmatrixPositive(self):
        ourSpline = self.spline
        N = ourSpline.Nmatrix()
        self.assertTrue(N.all() >= 0)
        
    def testXiPositive(self):
        sp = self.spline
        xi = sp.ma_version1()
        self.assertTrue(xi.all() >= 0)
        
if __name__ == '__main__':
    unittest.main()