//Author: Dr. Shantanu Shahane
#ifndef __coefficient_computations_H
#define __coefficient_computations_H
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "class.hpp"
#include <unistd.h>
#include <limits.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/MatOp/SparseGenRealShiftSolve.h>
using namespace std;

void shifting_scaling(vector<double> &vert, double *scale, int dim);

void shifting_scaling(double *xyz_interp, vector<double> &vert, double *scale, int dim);

void calc_PHS_RBF_grad_laplace_single_vert_A(vector<double> &vert, PARAMETERS &parameters, Eigen::MatrixXd &A, double *scale);

void calc_PHS_RBF_grad_laplace_single_vert_grad_x_rhs(vector<double> &vert, PARAMETERS &parameters, Eigen::MatrixXd &rhs, double *scale, vector<int> &central_vert_list);

void calc_PHS_RBF_interp_single_vert_rhs(double *xyz_interp, vector<double> &vert, PARAMETERS &parameters, Eigen::VectorXd &rhs);

void calc_PHS_RBF_grad_laplace_single_vert_grad_y_rhs(vector<double> &vert, PARAMETERS &parameters, Eigen::MatrixXd &rhs, double *scale, vector<int> &central_vert_list);

void calc_PHS_RBF_grad_laplace_single_vert_grad_z_rhs(vector<double> &vert, PARAMETERS &parameters, Eigen::MatrixXd &rhs, double *scale, vector<int> &central_vert_list);

void calc_PHS_RBF_grad_laplace_single_vert_laplacian_rhs(vector<double> &vert, PARAMETERS &parameters, Eigen::MatrixXd &rhs, double *scale, vector<int> &central_vert_list);

double calc_PHS_RBF_grad_laplace_single_vert(vector<double> &vert, PARAMETERS &parameters, Eigen::MatrixXd &laplacian, Eigen::MatrixXd &grad_x, Eigen::MatrixXd &grad_y, Eigen::MatrixXd &grad_z, double *scale, vector<int> &central_vert_list);

vector<vector<int>> calc_cloud_points_slow(vector<double> &xyz_probe, POINTS &points, PARAMETERS &parameters);

Eigen::SparseMatrix<double, Eigen::RowMajor> calc_interp_matrix(vector<double> &xyz_probe, POINTS &points, PARAMETERS &parameters);

#endif