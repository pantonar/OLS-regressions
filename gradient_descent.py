#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:38:16 2019

@author: pablo

Gradient descent for linear regression

"""

import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


 

directory=r'/Users/pablo/Desktop/edx/week7'
os.chdir(directory)

data=np.genfromtxt('input2.csv',delimiter=',')
n_tot=data.shape[1]
x=data[:,1:(n_tot)]
ones=np.ones((x.shape[0],x.shape[1]+1))
ones[:,1:]=x
x=ones
y=data[:,0]


 

class regression:
    ''' 
    needs to provide data (x), and labels (y)
    alpha : learning rate
    x needs to be shaped s.t. its first column are ones
    '''
    def __init__(self,x,y,alpha=1,epsilon=0.001):
        self.k=x.shape[1]
        self.n=x.shape[0]
        self.x_raw=x
        self.y=y
        self.betas=np.ones(self.k)
        self.alpha=alpha
        self.history=[self.betas]
        self.epsilon=epsilon
        
        # assign initial values
        self.Normalise()
        self.Gradient()
        self.Loss()
        
    def Loss(self):
        difference=self.y-self.x@self.betas
        self.loss=sum(np.square(difference))/(2*self.n)
        
    def Normalise(self):
        a=self.x_raw
        for i in range(1,self.k):
            a[:,i]=(x[:,i]-np.average(x[:,i]))/np.std(x[:,i])
        self.x=a
    
    def Gradient(self):
        x=self.x
        x_t=np.transpose(x)
        n=self.n
        y=self.y
        betas=self.betas
        self.gradient=(-x_t@y+(x_t@x)@betas)/n
          
    def GradientDescent(self):
        iters=0
        epsilon=1
        while epsilon>self.epsilon:
            if iters>=100:
                print('Convergence not achieved')
                break
            old_betas=self.betas
            self.betas=self.betas-self.alpha*self.gradient
            self.history.append(self.betas)
            self.Gradient()
            epsilon=sum(abs(old_betas-self.betas))
            iters+=1
        self.Loss()
        self.number_iterations=iters
                        
    def NormalEquation(self):
        x=self.x
        x_t=np.transpose(self.x)
        y=self.y
        x_t_x=np.matmul(x_t,x)
        x_t_y=np.matmul(x_t,y)
        self.beta_normal=np.matmul(np.linalg.inv(x_t_x),x_t_y)
        
    def Predict(self):
        self.y_hat=np.matmul(self.x,self.betas)
        
    def Visualise(self):
        self.Predict() # ensure y_hat is produced
        x_plot = self.x[:,1]
        y_plot = self.x[:,2]
        z_plot = self.y_hat

        # create canvas
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')       
        # shape data
        xx , yy = np.meshgrid(x_plot,y_plot)
        zz = np.ones(xx.shape)*self.betas[0] + self.betas[1]*xx + self.betas[2]*yy
        
        zy = self.y #data to plot datapoints
        # plot data
        ax.scatter(x_plot, y_plot, zy)
        ax.plot_surface(xx, yy, zz,alpha=0.2, color = [0,1,0]) #,linewidth=0, antialiased=False)
        
        plt.show() 

    
    
reg=regression(x,y,epsilon=0.00001) 
reg.loss  
reg.NormalEquation()
reg.GradientDescent()
#reg.Visualise()
reg.loss
reg.Predict()
y_hat=reg.y_hat

