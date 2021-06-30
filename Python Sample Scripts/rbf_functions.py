#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 08:57:29 2021

@author: shantanu
"""
import time
import pandas as pd
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures
from scipy.sparse import csr_matrix

class PARAMETERS:
    def __init__(self, polydeg, phsdeg, cloud_size_mult, meshfile):
        self.polydeg = polydeg; self.phsdeg = phsdeg; self.cloud_size_mult = cloud_size_mult
        self.meshfile = meshfile
        self.dim=None; self.num_poly_terms=None; self.cloud_size=None
        self.max_dx=None; self.min_dx=None; self.avg_dx=None

class POINTS:
    def __init__(self, parameters):
        data=pd.read_csv(parameters.meshfile);
        dim=2;
        for col in data.columns:
            if col=='z': dim=3; break;
        if dim==2:
            self.xyz=(data[['x','y']]).to_numpy();
            self.normal=(data[['normal_x','normal_y']]).to_numpy();
        else:
            self.xyz=(data[['x','y','z']]).to_numpy();
            self.normal=(data[['normal_x','normal_y','normal_z']]).to_numpy();
        self.boundary_flag=data['boundary_flag'].to_numpy().astype(bool);
        self.grad_x_mat=None; self.grad_y_mat=None; self.grad_z_mat=None; self.laplacian_mat=None;
        parameters.dim=dim; self.nv=self.xyz.shape[0]
        parameters.num_poly_terms=math.comb(parameters.polydeg + parameters.dim, parameters.dim)
        parameters.cloud_size = int(parameters.num_poly_terms * parameters.cloud_size_mult)

    def assemble_matrices(self, parameters, cloud):
        rows = np.ravel( np.transpose( np.tile(np.arange(self.nv), (parameters.cloud_size, 1)) ) )
        self.grad_x_mat = csr_matrix(( np.ravel(cloud.grad_x_coeff), ( rows, np.ravel(cloud.nb_pt) )), shape=(self.nv, self.nv))
        self.grad_y_mat = csr_matrix(( np.ravel(cloud.grad_y_coeff), ( rows, np.ravel(cloud.nb_pt) )), shape=(self.nv, self.nv))
        if parameters.dim == 3: self.grad_z_mat = csr_matrix(( np.ravel(cloud.grad_z_coeff), ( rows, np.ravel(cloud.nb_pt) )), shape=(self.nv, self.nv))
        self.laplacian_mat = csr_matrix(( np.ravel(cloud.laplacian_coeff), ( rows, np.ravel(cloud.nb_pt) )), shape=(self.nv, self.nv))



class CLOUD:
    def __init__(self, parameters, points):
        self.calc_cloud_points(parameters, points)
        self.calc_coeff(parameters, points)
        points.assemble_matrices(parameters, self)

    def calc_cloud_points(self, parameters, points):
        nbrs = NearestNeighbors(n_neighbors=parameters.cloud_size, algorithm='kd_tree').fit(points.xyz)
        self.nb_pt_dist, self.nb_pt = nbrs.kneighbors(points.xyz)
        xyz_temp=np.copy(points.xyz); #to couple boundary point only to interior points
        xyz_temp[points.boundary_flag,:]=1E5*np.max(np.abs(xyz_temp));# print(xyz_temp)
        nbrs = NearestNeighbors(n_neighbors=parameters.cloud_size-1, algorithm='kd_tree').fit(xyz_temp)
        self.nb_pt_dist[points.boundary_flag,1:], self.nb_pt[points.boundary_flag,1:] = nbrs.kneighbors(points.xyz[points.boundary_flag,:])
        parameters.max_dx=np.max(self.nb_pt_dist[:,1])
        parameters.min_dx=np.min(self.nb_pt_dist[:,1])
        parameters.avg_dx=np.mean(self.nb_pt_dist[:,1])

    def calc_coeff(self, parameters, points):
        assert parameters.phsdeg%2==1
        assert parameters.polydeg>1
        self.grad_x_coeff = np.zeros((self.nb_pt.shape))
        self.grad_y_coeff = np.zeros((self.nb_pt.shape))
        if parameters.dim==3: self.grad_z_coeff = np.zeros((self.nb_pt.shape))
        self.laplacian_coeff = np.zeros((self.nb_pt.shape))
        self.cond_num = np.zeros((self.nb_pt.shape[0],))
        timer_2=time.time(); timer_1=time.time();
        for iv in range(points.nv):
            coeff, self.cond_num[iv] = calc_coeffs_single_point(points.xyz[self.nb_pt[iv,:], :], self.nb_pt_dist[iv,:], parameters)
            self.grad_x_coeff[iv,:] = coeff[:,0]; self.grad_y_coeff[iv,:] = coeff[:,1]
            if parameters.dim==3: self.grad_z_coeff[iv,:] = coeff[:,2]
            self.laplacian_coeff[iv,:] = coeff[:,parameters.dim]

            if time.time()-timer_2>1:
                timer_2=time.time();
                print('    calc_coeff: Completed %i points out of %i (%.2g percent) in %g seconds' %( iv+1, points.nv, 100*(iv+1)/points.nv, time.time()-timer_1 ))

        self.cond_num_max = np.max(self.cond_num)
        self.cond_num_min = np.min(self.cond_num)
        self.cond_num_avg = np.mean(self.cond_num)

def calc_coeffs_single_point(xyz, dist, parameters):
    xyz=xyz-xyz[0,:]; dim=xyz.shape[1];
    z_pad=np.zeros((xyz.shape[0],xyz.shape[0]))
    x_pad=np.tile(xyz[:,0], (xyz.shape[0], 1)); y_pad=np.tile(xyz[:,1], (xyz.shape[0], 1))
    if dim==3: z_pad=np.tile(xyz[:,2], (xyz.shape[0], 1))
    A0 = ( (x_pad-np.transpose(x_pad))**2 + (y_pad-np.transpose(y_pad))**2 + (z_pad-np.transpose(z_pad))**2 ) **(parameters.phsdeg/2.0)
    P0 = PolynomialFeatures(parameters.polydeg).fit_transform(xyz)
    A = np.zeros((parameters.cloud_size + parameters.num_poly_terms, parameters.cloud_size + parameters.num_poly_terms))
    A[:parameters.cloud_size, :parameters.cloud_size] = A0
    A[:parameters.cloud_size, parameters.cloud_size:] = P0
    A[parameters.cloud_size:, :parameters.cloud_size] = np.transpose(P0)
    rhs=np.zeros((A.shape[0], dim+1))

    if dim==2:
        rhs[1:parameters.cloud_size,0] = parameters.phsdeg* xyz[1:,0] * (dist[1:]**(parameters.phsdeg-2.0))
        rhs[1:parameters.cloud_size,1] = parameters.phsdeg* xyz[1:,1] * (dist[1:]**(parameters.phsdeg-2.0))
        rhs[1:parameters.cloud_size,2] = (parameters.phsdeg**2) * (dist[1:]**(parameters.phsdeg-2.0))
        rhs[parameters.cloud_size+1,0]=1.0; rhs[parameters.cloud_size+2,1]=1.0;
        rhs[parameters.cloud_size+3,2]=2.0; rhs[parameters.cloud_size+5,2]=2.0;
    else:
        rhs[1:parameters.cloud_size,0] = parameters.phsdeg* xyz[1:,0] * (dist[1:]**(parameters.phsdeg-2.0))
        rhs[1:parameters.cloud_size,1] = parameters.phsdeg* xyz[1:,1] * (dist[1:]**(parameters.phsdeg-2.0))
        rhs[1:parameters.cloud_size,2] = parameters.phsdeg* xyz[1:,2] * (dist[1:]**(parameters.phsdeg-2.0))
        rhs[1:parameters.cloud_size,3] = ( (parameters.phsdeg**2) + parameters.phsdeg) * (dist[1:]**(parameters.phsdeg-2.0))
        rhs[parameters.cloud_size+1,0]=1.0; rhs[parameters.cloud_size+2,1]=1.0; rhs[parameters.cloud_size+3,2]=1.0;
        rhs[parameters.cloud_size+4,3]=2.0; rhs[parameters.cloud_size+7,3]=2.0; rhs[parameters.cloud_size+9,3]=2.0;

    coeffs=np.linalg.solve(A, rhs)[:xyz.shape[0], :];
    cond=np.linalg.cond(A)
    return coeffs, cond