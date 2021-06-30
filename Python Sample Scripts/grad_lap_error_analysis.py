#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 07:27:52 2021

@author: shantanu
"""
import time
import numpy as np
import rbf_functions as rbf
import general_functions as genf
import matplotlib.pyplot as plt
genf.tic()

wv=1.0 #wavenumber
folder='../gmsh_files/Square/'
meshfiles=['Square_n_10_unstruc_list.csv', 'Square_n_20_unstruc_list.csv', 'Square_n_30_unstruc_list.csv']

timer=time.time(); print()
l1_err_grad=np.zeros((1,len(meshfiles))); l1_err_lap=np.zeros((1,len(meshfiles)));
max_cond_num=np.zeros((1,len(meshfiles))); dx_list=[]; bandwidth_max=[]
for i1 in range(len(meshfiles)):
    print()
    parameters = rbf.PARAMETERS(polydeg=3, phsdeg=3, cloud_size_mult=2.0, meshfile=folder+meshfiles[i1])
    points = rbf.POINTS(parameters)
    cloud = rbf.CLOUD(parameters, points)
    dx_list.append(parameters.avg_dx)
    bandwidth_max.append( np.max( np.abs( cloud.nb_pt - np.reshape(cloud.nb_pt[:,0], (-1,1)) ) ) )

    func_ana = np.sin(wv*points.xyz[:,0]) * np.sin(wv*points.xyz[:,1])
    grad_x_ana = wv * np.cos(wv*points.xyz[:,0]) * np.sin(wv*points.xyz[:,1])
    grad_y_ana = wv * np.sin(wv*points.xyz[:,0]) * np.cos(wv*points.xyz[:,1])
    lap_ana = -wv*wv * 2.0 * func_ana

    grad_x_num = points.grad_x_mat.dot(func_ana);
    l1_err_grad[0,i1]=l1_err_grad[0,i1] + ( np.mean(np.abs(grad_x_num - grad_x_ana)) / 2.0 )
    grad_y_num = points.grad_y_mat.dot(func_ana)
    l1_err_grad[0,i1]=l1_err_grad[0,i1] + ( np.mean(np.abs(grad_y_num - grad_y_ana)) / 2.0 )
    lap_num = points.laplacian_mat.dot(func_ana)
    l1_err_lap[0,i1]= np.mean(np.abs(lap_num - lap_ana))
    max_cond_num[0,i1] = cloud.cond_num_max

    print('Completed %i meshfiles out of %i at %g seconds\n' %( i1+1, len(meshfiles), time.time()-timer ))

grad_slope = genf.bestfit_line_plot(dx_list, l1_err_grad, labels=['Gradient'], plot_flag=True)
lap_slope = genf.bestfit_line_plot(dx_list, l1_err_lap, labels=['Laplacian'], plot_flag=True)
cond_slope = genf.bestfit_line_plot(dx_list, max_cond_num, labels=['Max RBF Cond. No.'], plot_flag=True)

print('\n\nPolydeg: %i, Slopes gradient: %g, laplacian: %g, cond. no. : %g\n\n' %( parameters.polydeg, grad_slope, lap_slope, cond_slope ))

plt.figure(); plt.spy(points.grad_x_mat, marker='x', color='r'); plt.title('Sparsity Pattern')

genf.toc()