#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:07:58 2021

@author: shantanu
"""
import numpy as np
from scipy.sparse import csc_matrix

class scipy_iterative_solver_callback(object):
    def __init__(self, disp=True):
       self._disp = disp
       self.niter = 0
       self.callbacks = []
    def __call__(self, rk=None):
       self.callbacks.append(rk)
       self.niter += 1
#       if self._disp:
#           print('%s' %(str(rk)))

def poisson_matrix(parameters, points, cloud, dirichlet_flag):
    flag = np.logical_not(points.boundary_flag) #interior points
    rows = np.ravel( np.transpose( np.tile(np.arange(points.nv), (parameters.cloud_size, 1)) )[ flag, :] )
    cols = np.ravel(cloud.nb_pt[ flag, :])
    values = np.ravel(cloud.laplacian_coeff[ flag, :])

    flag = points.boundary_flag * dirichlet_flag #boundary points with dirichlet condition
    indices = np.where(flag)[0]; rows = np.append(rows, indices); cols = np.append(cols, indices)
    values = np.append(values, np.ones((len(indices),)))

    flag = points.boundary_flag * np.logical_not(dirichlet_flag) #boundary points with neumann condition
    assert np.sum(flag)==0

    matrix = csc_matrix(( values, ( rows, cols) ), shape=(points.nv, points.nv))
    return matrix