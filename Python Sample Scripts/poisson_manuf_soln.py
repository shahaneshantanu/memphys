#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:37:27 2021

@author: shantanu
"""
import time
import numpy as np
import rbf_functions as rbf
import general_functions as genf
import solver_functions as solf
import scipy.sparse.linalg as spla
genf.tic()

wv=1.0 #wavenumber
folder='../gmsh_files/Square/'
meshfiles=['Square_n_10_unstruc_list.csv', 'Square_n_20_unstruc_list.csv', 'Square_n_30_unstruc_list.csv']

timer=time.time(); print()
l1_err=np.zeros((1,len(meshfiles))); dx_list=[]; residual_list=[];
for i1 in range(len(meshfiles)):
    print()
    parameters = rbf.PARAMETERS(polydeg=3, phsdeg=3, cloud_size_mult=2.0, meshfile=folder+meshfiles[i1])
    points = rbf.POINTS(parameters)
    cloud = rbf.CLOUD(parameters, points)
    dx_list.append(parameters.avg_dx)

    dirichlet_flag = True * np.full((points.nv), True, dtype=bool)
    poisson_matrix = solf.poisson_matrix(parameters, points, cloud, dirichlet_flag)

    T_ana = np.sin(wv*points.xyz[:,0]) * np.sin(wv*points.xyz[:,1])
    rhs = -2*wv*wv * T_ana
    rhs[points.boundary_flag * dirichlet_flag] = T_ana[points.boundary_flag * dirichlet_flag] #dirichlet BC

    solver = spla.factorized(poisson_matrix)
    T_num = solver(rhs)

    l1_err[0,i1]= np.mean(np.abs(T_num - T_ana))

    print('Completed %i meshfiles out of %i at %g seconds\n' %( i1+1, len(meshfiles), time.time()-timer ))

err_slope = genf.bestfit_line_plot(dx_list, l1_err, labels=['Poisson Error'])
print('\n\nPolydeg: %i, Slope of Error: %g' %( parameters.polydeg, err_slope ))
print(); print(l1_err); print();


genf.toc()