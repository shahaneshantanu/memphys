//Author: Dr. Shantanu Shahane
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
#include "coefficient_computations.hpp"
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
#include "nanoflann.hpp"
using namespace std;

SOLIDIFICATION::SOLIDIFICATION(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, int temporal_order1)
{
    temporal_order = temporal_order1;
    if (temporal_order != 1 && temporal_order != 2)
    {
        printf("\n\nERROR from SOLIDIFICATION::SOLIDIFICATION temporal_order should be either '1' or '2'; current value: %i\n\n", temporal_order);
        throw bad_exception();
    }
    if (temporal_order == 2)
        T_old_old = Eigen::VectorXd::Zero(points.nv), fs_old_old = Eigen::VectorXd::Zero(points.nv);
    dfs_dT = Eigen::VectorXd::Zero(points.nv);
    T_source = Eigen::VectorXd::Zero(points.nv);
}

void SOLIDIFICATION::single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &T_new, Eigen::VectorXd &T_old, Eigen::VectorXd &fs_new, Eigen::VectorXd &fs_old, int it)
{
    if (temporal_order == 1 || it == 0)
    {
        T_source = T_old + (parameters.dt * alpha * (points.laplacian_matrix_EIGEN * T_old));
        T_source = T_source - ((Lf / Cp) * dfs_dT.cwiseProduct(T_old));
        T_source = T_source.cwiseQuotient(Eigen::VectorXd::Ones(points.nv) - ((Lf / Cp) * dfs_dT));
    }
    else
    {
        T_source = T_old + (parameters.dt * alpha * (points.laplacian_matrix_EIGEN * (1.5 * T_old - 0.5 * T_old_old)));
        T_source = T_source - ((1.5 * Lf / Cp) * dfs_dT.cwiseProduct(T_old));
        T_source = T_source + ((0.5 * Lf / Cp) * (3 * fs_old - 4 * fs_old + fs_old_old));
        T_source = T_source.cwiseQuotient(Eigen::VectorXd::Ones(points.nv) - ((1.5 * Lf / Cp) * dfs_dT));
    }

    for (int iv = 0; iv < points.nv; iv++)
        if (!points.boundary_flag[iv]) //boundary points should have dirichlet
            T_new[iv] = T_source[iv];  //non-boundary points only

    double fs_hat = 1.0 - pow((Tsol + Teps - Tf) / (Tliq - Tf), 1.0 / (k_partition - 1.0));
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (T_new[iv] <= Tsol)
            fs_new[iv] = 1.0;
        else if (T_new[iv] >= Tliq)
            fs_new[iv] = 0.0;
        else
            fs_new[iv] = 1.0 - pow((T_new[iv] - Tf) / (Tliq - Tf), 1.0 / (k_partition - 1.0));
        if ((T_new[iv] <= Tsol - Teps) || (T_new[iv] >= Tliq))
            dfs_dT[iv] = 0.0;
        else if ((T_new[iv] >= Tsol - Teps) && (T_new[iv] <= Tsol + Teps))
            dfs_dT[iv] = -(1.0 - fs_hat) / (2.0 * Teps);
        else
        {
            dfs_dT[iv] = pow((T_new[iv] - Tf) / (Tliq - Tf), (2.0 - k_partition) / (k_partition - 1.0));
            dfs_dT[iv] = -dfs_dT[iv] / ((k_partition - 1.0) * (Tliq - Tf));
        }
    }
    if (temporal_order == 2)
        T_old_old = T_old, fs_old_old = fs_old;
}