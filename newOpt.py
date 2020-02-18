# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:07:02 2019
@author: mans_
"""

from scipy import *
from pylab import *
from scipy import optimize
import matplotlib.pyplot as plt
from chebyquad_problem import *

class Optimization:
    """
    This is the Solver class for finding the minimum of a function.
    
    """
    
    def __init__(self, ProblemClass, xtol=10**(-9), h=10**(-8), eps=10**(-8), ra=0.1, 
                 sigma=0.7, tau=0.1, xi=9.):
        """
        INPUT:
            ProblemClass    - An object of the class 'Problem'
        
        OPTIONAL INPUT:
            xtol            - A scalar used to evaluate when to cancel the 
                              iteration of the call method
            h               - This is used in the grad method
            eps             - This is used in the hessian method
            ra              - Is used in the condition method
            sigma           - Is used in the condition method
            tau             - Is used in block1 and block2 methods
            xi              - Is used in block1 method
            
        """
        
        self.ProblemClass= ProblemClass
        self.func = ProblemClass.func
        self.x0 = ProblemClass.x0
        self.xx = ProblemClass.x0
        
        self.xtol = xtol 
        self.h = h
        self.eps = eps
        self.ra = ra
        self.sigma = sigma # Needs to be greater than ra
        self.tau = tau 
        self.xi = xi
        
    def __call__(self, method='exact', newton = 'classical'):
        """
        This is the method that performs the iteration to obtain xnew. 
        ------
        INPUT:
            method      - exact, g (Goldstein) , w (Wolfe). These are used to 
                          obtain tha alpha.
            newton      - classical, goodbroyden, badbroyden, DFP2, BFG2. These
                          these are used to get the direction of the step.
            
        OUTPUT:
            xnew        - If we converge we return a list with the minima. If 
                          not we return None.
        """
        x0 = self.x0
        xold=x0
        xolder=[0.95*x for x in x0] 
        try:
            Hold = identity(len(x0)) 
        except TypeError:
            Hold = 1
        Hk = self.broyden(xold,xolder,Hold)
        for i in range(0,1000):
            if self.ProblemClass.CheckGrad(): # Checking if the user has already added a grad function in the problem class
                g = self.ProblemClass.grad(xold)
            else:
                g = self.grad(xold, self.func) 
                
            if newton == 'classical':
                if self.ProblemClass.CheckHessian(): # Checking if the user has already added a Hessian in the problem class
                    G = self.ProblemClass.Hessian
                else:
                    G = self.hessian(xold)
                s = solve(G,g)
            elif newton == 'goodbroyden':
                Q = self.broyden(xold, xolder, Hold, 'good')
                Hold=Q
                s = solve(Q,g)
                xolder = xold
            elif newton == 'badbroyden':
                H = self.broyden(xold, xolder, Hold, 'bad')
                Hold=H
                s = H@g
                xolder = xold
            elif newton == 'DFP2' :
                H = self.DFP2(xold, xolder, Hk)
                Hold=H
                s = H@g
                xolder = xold
            elif newton == 'BFG2':
                H = self.BFG2(xold, xolder, Hk)
                Hold=H
                s = H@g
                xolder = xold
                
            fa = lambda a: self.func(xold - a*s)
            alpha = self.linesearch(fa, method)
            
            xnew = xold-alpha*s
            self.xx = append(self.xx,xnew)
            
            if norm(xold-xnew) < self.xtol:
                return xnew
            xold = xnew
        return None
         
    def grad(self,x, func):
        """
        This method returns the gradient approximation at a certain point.
        -----
        INPUT: 
            x       - The point that we want evaluated.
            func    - The function that we seek the gradient of.
        OUTPUT: 
            The gradient value at point x
        """
        try:
            f = zeros(len(x))
            for i in range(len(x)):
                ei = zeros(len(x))
                ei[i] = self.h
                f[i] = (func(x+ei) - func(x))/self.h
            return f
        except TypeError:
            return (func(x+self.h) - func(x))/self.h
     
    
    def hessian(self,x):
        """
        This method returns the Hessian approximation at a certain point, 
        i.e. the second derivative of the function.
        -----
        INPUT: 
            x       - The point that we want evaluated.
            
        OUTPUT: 
            hessian - The Hessian matrix value at point x
        """
        hessian = zeros((len(x),len(x)))
        func = self.func
        
        for i in range(len(x)):     # 0, 1
            ei = zeros(len(x))
            ei[i] = self.eps
            
            for j in range(len(x)): # 0, 1
                ej = zeros(len(x))
                ej[j] = self.eps
                hessian[i,j] = (func(x+ei+ej) - func(x+ei) - func(x+ej) + func(x))/self.eps**2
        
        return hessian
    
    def broyden(self, xold, xolder, Qold, quality = 'bad'):
        """
        
        -----
        INPUT: 
            xold            - x at point k
            xolder          - x at point k-1
            Qold            - Matrix
        
        OPTIONAL INPUT:    
            quality         - good, bad
            
            
        OUTPUT: 
            Q               - 
            H               -
        """
        
        delta = array(xold) - array(xolder)
        gamma = self.grad(xold, self.func) - self.grad(xolder, self.func)
        if quality == 'good':
            v = (gamma - Qold@delta)/(delta.T@delta)
            w = delta
            Q = Qold + v@w.T
            return (Q + Q.T)/2
        elif quality == 'bad':
            Hold = Qold
            u = delta - Hold@gamma
            a = 1/(u.T@gamma)
            H = Hold + a*u@u.T
            return (H + H.T)/2
    
    def DFP2(self,xold,xolder,Hk):
        """
        
        -----
        INPUT: 
            xold            - x at point k
            xolder          - x at point k-1
            Hk              - The H-matrix at point k    
            
        OUTPUT: 
            H               - The H-matrix at point k+1
        """
       delta = array(xold) - array(xolder)
       gamma = self.grad(xold, self.func) - self.grad(xolder, self.func)
       H = Hk + ((delta@delta.T)/(delta.T@gamma)) -(Hk*gamma@gamma.T*Hk)/(gamma.T*Hk*gamma)
       return (H+H.T)/2
       
    def BFG2(self,xold,xolder,Hk):
       delta = array(xold) - array(xolder)
       gamma = self.grad(xold, self.func) - self.grad(xolder, self.func)
       B= (1+(gamma.T*Hk*gamma)/(delta.T@gamma))*(delta@delta.T)/(delta.T@gamma)
       C = ((delta*gamma.T*Hk+Hk*gamma*delta.T))/(delta.T@gamma)
       H = Hk+B-C 
       return H
        
    def condition(self, method, fa, a0, aU, aL):
        """
        This is a method to calculate the left and right conditions. 
        -----
        INPUT: 
            method          - exact, g (Goldstein) , w (Wolfe)             
            fa              - a function of alpha
            a0              - 
            aU              -
            aL              -
            
        OUTPUT: 
            LC              - A boolean for the left condition
            RC              - A boolean for the right condition
        """
        ra = self.ra
        sigma = self.sigma
        
        dfaL = self.grad(aL,fa) # This is f_a prime of aL
        print(dfaL)
        if method=='g':
            LC = (fa(a0) >= fa(aL) + (1-ra)*(a0-aL)*dfaL)
            RC = (fa(a0) <= fa(aL) + ra*(a0-aL)*dfaL)
        elif method=='w':
            LC = (self.grad(a0,fa) >= sigma*dfaL)
            RC = (fa(a0) <= fa(aL) + ra*(a0-aL)*dfaL)
        return [LC, RC]
    
    
    def linesearch(self, fa, method):
        if method=='exact':
            a0 = optimize.fmin(fa,0.5, disp = 0) # gör tyst
            return a0

        a0 = 5.0
        aL = 0.
        aU = 1.e99
        
        [LC, RC] = self.condition(method, fa, a0, aU, aL)
        
        k = 0
        
        while not (LC and RC):
            if  not LC:
                [a0, aU, aL] = self.block1(a0, aU, aL, fa)
                
            else:
                [a0, aU, aL] = self.block2(a0, aU, aL, fa)
            [LC, RC] = self.condition(method, fa, a0, aU, aL)
            print(k)
            k = k+1
            
        return a0
        
    
    def block1(self, a0, aU, aL, fa):
        tau = self.tau
        xi = self.xi
        
        delta_a0 = self.extrapol(a0, aU, aL, fa)
        delta_a0 = max(delta_a0, tau*(a0-aL))
        delta_a0 = min(delta_a0, xi*(a0-aL))
        aL = a0
        a0 = a0 + delta_a0
        return [a0, aU, aL]
    
    def block2(self, a0, aU, aL, fa):
        tau = self.tau
        
        aU = min(a0,aU)
        bar_a0 = self.interpol(a0, aU, aL, fa)
        bar_a0 = max(bar_a0, aL + tau*(aU-aL))
        bar_a0 = min(bar_a0, aU - tau*(aU-aL))
        a0 = bar_a0
        return [a0, aU, aL]
    
    def extrapol(self, a0, aU, aL, fa):
#        print(aL)
#        print(a0)
#        print(fa)
#        print(self.grad(aL,fa) - self.grad(a0,fa))
#        if a0 == aL:
#            return 0
        return (a0-aL)*self.grad(a0,fa)/(self.grad(aL,fa) - self.grad(a0,fa))
    
    def interpol(self, a0, aU, aL, fa):
        return (a0-aL)**2*self.grad(aL,fa)/(2*(fa(aL) - fa(a0) + (a0-aL)*self.grad(aL,fa)))
        
    def plot(self):
        x1 = linspace(-0.5,2,1000)
        x2 = linspace(-1.5,4,1000)
        X1, X2 = meshgrid(x1, x2)
        plt.figure()
        #print(self.func((X1,X2)))
        cp = plt.contour(X1,X2,self.func((X1,X2)),[0,1,3,10,50,100,500,800],colors='black')
        plt.clabel(cp, inline=True, fontsize=10)
        plt.plot(self.xx[0::2],self.xx[1::2],'o',color='black')
        plt.plot(self.xx[-1],self.xx[-2],'o',color='red')
        plt.show()
        pass      
    
class Problem:
    """
    This is the problem class that will later be used in our solver class (Optimizer)
    """
    def __init__(self,func,x0,grad=None, Hessian=None):
        self.func = func
        self.x0 = x0
        self.grad = grad
        self.Hessian = Hessian
        
    def CheckGrad(self):
        if self.grad is None:
            return False
        else:
            return True
    
    def CheckHessian(self):
        if self.Hessian is None:
            return False
        else:
            return True
    
    
def f(x):
    return 100*(x[1]-x[0]**2)**(2) + (1-x[0])**2
def easy(x):
    return (x[1]-2)**2 + (1-x[0])**2

P = Problem(f,[0.7,0.7])
O = Optimization(P) 
#H_matrix = O.Hessian(array([0.4,0.6]))         
x = O(method = 'g', newton = 'classical')
O.plot()