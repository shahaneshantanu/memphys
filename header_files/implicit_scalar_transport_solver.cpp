//Author: Dr. Shantanu Shahane
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>
#include <Eigen/Core>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "general_functions.hpp"
#include "class.hpp"

IMPLICIT_SCALAR_TRANSPORT_SOLVER::IMPLICIT_SCALAR_TRANSPORT_SOLVER(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &dirichlet_flag1, int precond_freq_it1, double unsteady_coeff1, double conv_coeff1, double diff_coeff1, bool solver_log_flag1)
{
    dirichlet_flag = dirichlet_flag1, solver_log_flag = solver_log_flag1;
    unsteady_coeff = unsteady_coeff1, conv_coeff = conv_coeff1, diff_coeff = diff_coeff1;
    precond_freq_it = precond_freq_it1;

    zero_vector = Eigen::VectorXd::Zero(points.nv);
    bc_full_neumann = true;
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv] && dirichlet_flag[iv])
        { //boundary point found with dirichlet BC
            bc_full_neumann = false;
            break;
        }
    if (bc_full_neumann)
    {
        printf("\n\nERROR from IMPLICIT_SCALAR_TRANSPORT_SOLVER::IMPLICIT_SCALAR_TRANSPORT_SOLVER Setting  full Neumann BC is not permitted\n\n");
        throw bad_exception();
    }
    source = zero_vector, phi_old_old = zero_vector;
}

void IMPLICIT_SCALAR_TRANSPORT_SOLVER::set_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new)
{
    matrix.resize(0, 0);
    vector<Eigen::Triplet<double>> triplet;
    int ivnb, dim = parameters.dimension;
    double value, unsteady_factor;
    if (it == 0)
        unsteady_factor = 1.0; //BDF1: implicit Euler
    else
        unsteady_factor = bdf2_alpha_1;
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (points.boundary_flag[iv])
        {
            if (dirichlet_flag[iv])
                triplet.push_back(Eigen::Triplet<double>(iv, iv, 1.0));
            else
            {
                for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                {
                    ivnb = cloud.nb_points_col[i1];
                    value = points.normal[dim * iv] * cloud.grad_x_coeff[i1] + points.normal[dim * iv + 1] * cloud.grad_y_coeff[i1];
                    triplet.push_back(Eigen::Triplet<double>(iv, ivnb, value));
                }
            }
        }
        else
        {
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                ivnb = cloud.nb_points_col[i1];
                value = diff_coeff * cloud.laplacian_coeff[i1];                    //diffusion
                value = value + (u_new[iv] * conv_coeff * cloud.grad_x_coeff[i1]); //convection
                value = value + (v_new[iv] * conv_coeff * cloud.grad_y_coeff[i1]); //convection
                if (ivnb == iv)
                    value = value + (unsteady_factor * unsteady_coeff); //diagonal term
                triplet.push_back(Eigen::Triplet<double>(iv, ivnb, value));
            }
        }
    }
    matrix.resize(points.nv, points.nv);
    matrix.setFromTriplets(triplet.begin(), triplet.end());
    matrix.makeCompressed();
    triplet.clear();
}

void IMPLICIT_SCALAR_TRANSPORT_SOLVER::modify_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new)
{
    int ivnb, dim = parameters.dimension, index;
    double value, unsteady_factor;
    if (it == 0)
        unsteady_factor = 1.0; //BDF1: implicit Euler
    else
        unsteady_factor = bdf2_alpha_1;
    for (int iv = 0; iv < points.nv; iv++)
        if (!points.boundary_flag[iv])
        { //coefficients of boundary points never updated for velocities
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                ivnb = cloud.nb_points_col[i1];
                value = diff_coeff * cloud.laplacian_coeff[i1];                    //diffusion
                value = value + (u_new[iv] * conv_coeff * cloud.grad_x_coeff[i1]); //convection
                value = value + (v_new[iv] * conv_coeff * cloud.grad_y_coeff[i1]); //convection
                if (ivnb == iv)
                    value = value + (unsteady_factor * unsteady_coeff); //diagonal term
                index = nb_points_col_matrix[i1];
                matrix.valuePtr()[index] = value;
            }
        }
}

void IMPLICIT_SCALAR_TRANSPORT_SOLVER::calc_nb_points_col_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters)
{
    int ivnb, index;
    nb_points_col_matrix.clear();
    for (int i1 = 0; i1 < cloud.nb_points_col.size(); i1++) //initialize to -1
        nb_points_col_matrix.push_back(-1);
    for (int iv = 0; iv < points.nv; iv++)
        if (!points.boundary_flag[iv])
        { //coefficients of boundary points never updated for velocities; nb_points_col_matrix ahas value of [-1] at bounday points
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                index = -1;
                ivnb = cloud.nb_points_col[i1];
                for (int i2 = matrix.outerIndexPtr()[iv]; i2 < matrix.outerIndexPtr()[iv + 1]; i2++)
                    if (matrix.innerIndexPtr()[i2] == ivnb)
                    {
                        index = i2;
                        break;
                    }
                if (index < 0)
                {
                    cout << "\n\nError from SEMI_IMPLICIT_SPLIT_SOLVER::calc_nb_points_col_matrix in matrix_u, unable to find ivnb: " << ivnb << " for iv: " << iv << ", points.boundary_flag[iv]: " << points.boundary_flag[iv] << "\n\n";
                    throw bad_exception();
                }
                nb_points_col_matrix[i1] = index;
            }
        }
}

void IMPLICIT_SCALAR_TRANSPORT_SOLVER::single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &phi_new, Eigen::VectorXd &phi_old, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, int it1)
{
    it = it1;
    if (it % precond_freq_it == 0 || it == 0 || it == 1)
    {
        set_matrix(points, cloud, parameters, u_new, v_new);
        solver_eigen.setTolerance(parameters.solver_tolerance); //default is machine precision (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#ac160a444af8998f93da9aa30e858470d)
        solver_eigen.setMaxIterations(parameters.n_iter);       //default is twice number of columns (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#af83de7a7d31d9d4bd1fef6222b07335b)
        solver_eigen.preconditioner().setDroptol(parameters.precond_droptol);
        solver_eigen.compute(matrix);
        if (it == 0)
            calc_nb_points_col_matrix(points, cloud, parameters);
    }
    modify_matrix(points, cloud, parameters, u_new, v_new);

    if (it == 0) //BDF1: implicit Euler
        source = (unsteady_coeff * phi_old);
    else //BDF2
        source = -((bdf2_alpha_2 * unsteady_coeff) * phi_old) - ((bdf2_alpha_3 * unsteady_coeff) * phi_old_old);

    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
        { //retain boundary condition from "_old"
            if (dirichlet_flag[iv])
                source[iv] = phi_old[iv];
            else
                source[iv] = 0.0;
        }
    phi_new = solver_eigen.solveWithGuess(source, phi_old);
    if (solver_log_flag)
    {
        double absolute_residual = (matrix * phi_new - source).norm();
        double relative_residual = absolute_residual / source.norm();
        parameters.rel_res_log.push_back(relative_residual);
        parameters.abs_res_log.push_back(absolute_residual);
        parameters.n_iter_actual.push_back(solver_eigen.iterations());
    }
    phi_old_old = phi_old;
}