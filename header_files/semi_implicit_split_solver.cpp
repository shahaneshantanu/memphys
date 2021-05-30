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

SEMI_IMPLICIT_SPLIT_SOLVER::SEMI_IMPLICIT_SPLIT_SOLVER(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &u_dirichlet_flag1, vector<bool> &v_dirichlet_flag1, vector<bool> &p_dirichlet_flag1, int n_outer_iter1, double iterative_tolerance1, int precond_freq_it1)
{
    u_dirichlet_flag = u_dirichlet_flag1, v_dirichlet_flag = v_dirichlet_flag1, p_dirichlet_flag = p_dirichlet_flag1;
    n_outer_iter = n_outer_iter1, iterative_tolerance = iterative_tolerance1, precond_freq_it = precond_freq_it1;
    check_bc(points, parameters);
    clock_t clock_t1 = clock(), clock_t2 = clock();
    solver_p.init(points, cloud, parameters, p_dirichlet_flag, 0.0, 0.0, 1.0, true); //BC for p_prime is identical to BC of p_new
    parameters.factoring_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;

    zero_vector = Eigen::VectorXd::Zero(points.nv);
    zero_vector_1 = Eigen::VectorXd::Zero(points.nv + 1);
    p_bc_full_neumann = true;
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv] && p_dirichlet_flag[iv])
        { //boundary point found with dirichlet BC
            p_bc_full_neumann = false;
            break;
        }

    if (p_bc_full_neumann)
        p_source = zero_vector_1, p_prime = zero_vector_1; //this is full Neumann for pressure
    else
        p_source = zero_vector, p_prime = zero_vector;
    u_source = zero_vector, v_source = zero_vector;
    u_old_old = zero_vector, v_old_old = zero_vector;
    u_prime = zero_vector, v_prime = zero_vector;
    u_iter_old = zero_vector, v_iter_old = zero_vector;
    normal_mom_x = zero_vector, normal_mom_y = zero_vector;
}

void SEMI_IMPLICIT_SPLIT_SOLVER::check_bc(POINTS &points, PARAMETERS &parameters)
{
    int u_dirichlet_flag_sum = accumulate(u_dirichlet_flag.begin(), u_dirichlet_flag.end(), 0);
    if (u_dirichlet_flag_sum == 0)
    {
        printf("\n\nERROR from SEMI_IMPLICIT_SPLIT_SOLVER::check_bc Setting u_dirichlet_flag to full Neumann BC is not permitted; sum of u_dirichlet_flag: %i\n\n", u_dirichlet_flag_sum);
        throw bad_exception();
    }
    int v_dirichlet_flag_sum = accumulate(v_dirichlet_flag.begin(), v_dirichlet_flag.end(), 0);
    if (v_dirichlet_flag_sum == 0)
    {
        printf("\n\nERROR from FRACTIONASEMI_IMPLICIT_SPLIT_SOLVERL_STEP_1::check_bc Setting v_dirichlet_flag to full Neumann BC is not permitted; sum of v_dirichlet_flag: %i\n\n", v_dirichlet_flag_sum);
        throw bad_exception();
    }
    if (parameters.rho < 0 || parameters.mu < 0)
    {
        printf("\n\nERROR from SEMI_IMPLICIT_SPLIT_SOLVER::check_bc Some parameters are not set; parameters.rho: %g, parameters.mu: %g\n\n", parameters.rho, parameters.mu);
        throw bad_exception();
    }
}

void SEMI_IMPLICIT_SPLIT_SOLVER::calc_vel(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y)
{
    if (it == 0)
    { //BDF1: implicit Euler
        u_source = ((parameters.rho / parameters.dt) * u_old) - (points.grad_x_matrix_EIGEN_internal * p_new.head(points.nv)) + body_force_x;
        v_source = ((parameters.rho / parameters.dt) * v_old) - (points.grad_y_matrix_EIGEN_internal * p_new.head(points.nv)) + body_force_y;
    }
    else
    { //BDF2
        u_source = -((bdf2_alpha_2 * parameters.rho / parameters.dt) * u_old) - ((bdf2_alpha_3 * parameters.rho / parameters.dt) * u_old_old) - (points.grad_x_matrix_EIGEN_internal * p_new.head(points.nv)) + body_force_x;
        v_source = -((bdf2_alpha_2 * parameters.rho / parameters.dt) * v_old) - ((bdf2_alpha_3 * parameters.rho / parameters.dt) * v_old_old) - (points.grad_y_matrix_EIGEN_internal * p_new.head(points.nv)) + body_force_y;
    }

    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
        { //retain boundary condition from "_old"
            if (u_dirichlet_flag[iv])
                u_source[iv] = u_old[iv];
            else
                u_source[iv] = 0.0;
            if (v_dirichlet_flag[iv])
                v_source[iv] = v_old[iv];
            else
                v_source[iv] = 0.0;
        }
    u_new = solver_eigen_u.solveWithGuess(u_source, u_new);
    v_new = solver_eigen_v.solveWithGuess(v_source, v_new);
}

void SEMI_IMPLICIT_SPLIT_SOLVER::calc_vel(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
{
    if (it == 0)
    { //BDF1: implicit Euler
        u_source = ((parameters.rho / parameters.dt) * u_old) - (points.grad_x_matrix_EIGEN_internal * p_new.head(points.nv));
        v_source = ((parameters.rho / parameters.dt) * v_old) - (points.grad_y_matrix_EIGEN_internal * p_new.head(points.nv));
    }
    else
    { //BDF2
        u_source = -((bdf2_alpha_2 * parameters.rho / parameters.dt) * u_old) - ((bdf2_alpha_3 * parameters.rho / parameters.dt) * u_old_old) - (points.grad_x_matrix_EIGEN_internal * p_new.head(points.nv));
        v_source = -((bdf2_alpha_2 * parameters.rho / parameters.dt) * v_old) - ((bdf2_alpha_3 * parameters.rho / parameters.dt) * v_old_old) - (points.grad_y_matrix_EIGEN_internal * p_new.head(points.nv));
    }

    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
        { //retain boundary condition from "_old"
            if (u_dirichlet_flag[iv])
                u_source[iv] = u_old[iv];
            else
                u_source[iv] = 0.0;
            if (v_dirichlet_flag[iv])
                v_source[iv] = v_old[iv];
            else
                v_source[iv] = 0.0;
        }
    u_new = solver_eigen_u.solveWithGuess(u_source, u_new);
    v_new = solver_eigen_v.solveWithGuess(v_source, v_new);
}

void SEMI_IMPLICIT_SPLIT_SOLVER::calc_pressure(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
{
    if (p_bc_full_neumann)
        p_source = zero_vector_1; //this is full Neumann for pressure
    else
        p_source = zero_vector;
    if (it == 0) //BDF1: implicit Euler
        p_source.head(points.nv) = ((points.grad_x_matrix_EIGEN_internal * u_new) + (points.grad_y_matrix_EIGEN_internal * v_new)) * (parameters.rho / parameters.dt);
    else
        p_source.head(points.nv) = ((points.grad_x_matrix_EIGEN_internal * u_new) + (points.grad_y_matrix_EIGEN_internal * v_new)) * (bdf2_alpha_1 * parameters.rho / parameters.dt);
    //BC for p_prime is identical to BC of p_new
    //p_source of p_prime is zero for both dirichlet and neumann for p_new
    if (p_bc_full_neumann) //this is full Neumann for pressure
        solver_p.general_solve(points, parameters, p_prime, zero_vector_1, p_source);
    else
        solver_p.general_solve(points, parameters, p_prime, zero_vector, p_source);
    p_new = p_new + p_prime;

    normal_mom_x = zero_vector, normal_mom_y = zero_vector;
    if (it == 0)
    { //BDF1: implicit Euler
        normal_mom_x = -parameters.rho * (u_new.cwiseProduct(points.grad_x_matrix_EIGEN_boundary * u_new) + v_new.cwiseProduct(points.grad_y_matrix_EIGEN_boundary * u_new)) + parameters.mu * (points.laplacian_matrix_EIGEN_boundary * u_new) - (parameters.rho / parameters.dt) * (u_new - u_old);
        normal_mom_y = -parameters.rho * (u_new.cwiseProduct(points.grad_x_matrix_EIGEN_boundary * v_new) + v_new.cwiseProduct(points.grad_y_matrix_EIGEN_boundary * v_new)) + parameters.mu * (points.laplacian_matrix_EIGEN_boundary * v_new) - (parameters.rho / parameters.dt) * (v_new - v_old);
    }
    else
    {
        normal_mom_x = -parameters.rho * (u_new.cwiseProduct(points.grad_x_matrix_EIGEN_boundary * u_new) + v_new.cwiseProduct(points.grad_y_matrix_EIGEN_boundary * u_new)) + parameters.mu * (points.laplacian_matrix_EIGEN_boundary * u_new) - (parameters.rho / parameters.dt) * ((bdf2_alpha_1 * u_new) + (bdf2_alpha_2 * u_old) + (bdf2_alpha_3 * u_old_old));
        normal_mom_y = -parameters.rho * (u_new.cwiseProduct(points.grad_x_matrix_EIGEN_boundary * v_new) + v_new.cwiseProduct(points.grad_y_matrix_EIGEN_boundary * v_new)) + parameters.mu * (points.laplacian_matrix_EIGEN_boundary * v_new) - (parameters.rho / parameters.dt) * ((bdf2_alpha_1 * v_new) + (bdf2_alpha_2 * v_old) + (bdf2_alpha_3 * v_old_old));
    }

    double rhs, diag_coeff, off_diag_coeff;
    int dim = parameters.dimension, ivnb;

    for (int iv = 0; iv < points.nv; iv++)
    { //set boundary condition
        if (points.boundary_flag[iv])
        {
            if (!p_dirichlet_flag[iv]) //p_new for dirichlet BC is retained when p_prime is added (p_prime is zero at dirichlet BC)
            {                          //p_new for neumann BC has to be set explicitly
                rhs = (normal_mom_x[iv] * points.normal[dim * iv]) + (normal_mom_y[iv] * points.normal[dim * iv + 1]);
                for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                {
                    ivnb = cloud.nb_points_col[i1];
                    if (ivnb == iv)
                        diag_coeff = (cloud.grad_x_coeff[i1] * points.normal[dim * iv]) + (cloud.grad_y_coeff[i1] * points.normal[dim * iv + 1]);
                    else
                    {
                        off_diag_coeff = (cloud.grad_x_coeff[i1] * points.normal[dim * iv]) + (cloud.grad_y_coeff[i1] * points.normal[dim * iv + 1]);
                        rhs = rhs - (off_diag_coeff * p_new[ivnb]);
                    }
                }
                p_new[iv] = rhs / diag_coeff;
            }
        }
    }
}

void SEMI_IMPLICIT_SPLIT_SOLVER::calc_vel_corr(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new)
{
    if (it == 0)
    { //BDF1: implicit Euler
        u_prime = ((points.grad_x_matrix_EIGEN_internal * p_prime.head(points.nv)) * (parameters.dt / parameters.rho));
        v_prime = ((points.grad_y_matrix_EIGEN_internal * p_prime.head(points.nv)) * (parameters.dt / parameters.rho));
    }
    else
    {
        u_prime = ((points.grad_x_matrix_EIGEN_internal * p_prime.head(points.nv)) * (parameters.dt / (bdf2_alpha_1 * parameters.rho)));
        v_prime = ((points.grad_y_matrix_EIGEN_internal * p_prime.head(points.nv)) * (parameters.dt / (bdf2_alpha_1 * parameters.rho)));
    }

    u_new = u_new - u_prime;
    v_new = v_new - v_prime;
    //internal points are set above; boundary values are obtained implicitly in the solver (for both dirichlet and neumann); u_prime and v_prime are zero at all boundaries
}

void SEMI_IMPLICIT_SPLIT_SOLVER::extras(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
{
    // double total_steady_err, max_err, l1_err;
    // calc_max_l1_error(u_old, u_new, max_err, l1_err);
    // total_steady_err = l1_err / parameters.dt;
    // calc_max_l1_error(v_old, v_new, max_err, l1_err);
    // total_steady_err += l1_err / parameters.dt;
    // parameters.steady_error_log.push_back(total_steady_err);
    u_old_old = u_old, v_old_old = v_old;
    // return total_steady_err;
}

void SEMI_IMPLICIT_SPLIT_SOLVER::modify_vel_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new)
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
                value = -parameters.mu * cloud.laplacian_coeff[i1];                    //diffusion
                value = value + (u_new[iv] * parameters.rho * cloud.grad_x_coeff[i1]); //convection
                value = value + (v_new[iv] * parameters.rho * cloud.grad_y_coeff[i1]); //convection
                if (ivnb == iv)
                    value = value + (unsteady_factor * parameters.rho / parameters.dt); //diagonal term
                index = nb_points_col_matrix_u[i1];
                matrix_u.valuePtr()[index] = value;
                index = nb_points_col_matrix_v[i1];
                matrix_v.valuePtr()[index] = value;
            }
        }
}

void SEMI_IMPLICIT_SPLIT_SOLVER::set_vel_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new)
{
    matrix_u.resize(0, 0), matrix_v.resize(0, 0);
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
            if (u_dirichlet_flag[iv])
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
                value = -parameters.mu * cloud.laplacian_coeff[i1];                    //diffusion
                value = value + (u_new[iv] * parameters.rho * cloud.grad_x_coeff[i1]); //convection
                value = value + (v_new[iv] * parameters.rho * cloud.grad_y_coeff[i1]); //convection
                if (ivnb == iv)
                    value = value + (unsteady_factor * parameters.rho / parameters.dt); //diagonal term
                triplet.push_back(Eigen::Triplet<double>(iv, ivnb, value));
            }
        }
    }
    matrix_u.resize(points.nv, points.nv);
    matrix_u.setFromTriplets(triplet.begin(), triplet.end());
    matrix_u.makeCompressed();
    triplet.clear();

    for (int iv = 0; iv < points.nv; iv++)
    {
        if (points.boundary_flag[iv])
        {
            if (v_dirichlet_flag[iv])
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
                value = -parameters.mu * cloud.laplacian_coeff[i1];                    //diffusion
                value = value + (u_new[iv] * parameters.rho * cloud.grad_x_coeff[i1]); //convection
                value = value + (v_new[iv] * parameters.rho * cloud.grad_y_coeff[i1]); //convection
                if (ivnb == iv)
                    value = value + (parameters.rho / parameters.dt); //diagonal term
                triplet.push_back(Eigen::Triplet<double>(iv, ivnb, value));
            }
        }
    }
    matrix_v.resize(points.nv, points.nv);
    matrix_v.setFromTriplets(triplet.begin(), triplet.end());
    matrix_v.makeCompressed();
    triplet.clear();
}

void SEMI_IMPLICIT_SPLIT_SOLVER::calc_nb_points_col_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters)
{
    int ivnb, index;
    nb_points_col_matrix_u.clear(), nb_points_col_matrix_v.clear();
    for (int i1 = 0; i1 < cloud.nb_points_col.size(); i1++) //initialize to -1
        nb_points_col_matrix_u.push_back(-1), nb_points_col_matrix_v.push_back(-1);
    for (int iv = 0; iv < points.nv; iv++)
        if (!points.boundary_flag[iv])
        { //coefficients of boundary points never updated for velocities; nb_points_col_matrix ahas value of [-1] at bounday points
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                index = -1;
                ivnb = cloud.nb_points_col[i1];
                for (int i2 = matrix_u.outerIndexPtr()[iv]; i2 < matrix_u.outerIndexPtr()[iv + 1]; i2++)
                    if (matrix_u.innerIndexPtr()[i2] == ivnb)
                    {
                        index = i2;
                        break;
                    }
                if (index < 0)
                {
                    cout << "\n\nError from SEMI_IMPLICIT_SPLIT_SOLVER::calc_nb_points_col_matrix in matrix_u, unable to find ivnb: " << ivnb << " for iv: " << iv << ", points.boundary_flag[iv]: " << points.boundary_flag[iv] << "\n\n";
                    throw bad_exception();
                }
                nb_points_col_matrix_u[i1] = index;
            }
        }
    for (int iv = 0; iv < points.nv; iv++)
        if (!points.boundary_flag[iv])
        { //coefficients of boundary points never updated for velocities; nb_points_col_matrix ahas value of [-1] at bounday points
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                index = -1;
                ivnb = cloud.nb_points_col[i1];
                for (int i2 = matrix_v.outerIndexPtr()[iv]; i2 < matrix_v.outerIndexPtr()[iv + 1]; i2++)
                    if (matrix_v.innerIndexPtr()[i2] == ivnb)
                    {
                        index = i2;
                        break;
                    }
                if (index < 0)
                {
                    cout << "\n\nError from SEMI_IMPLICIT_SPLIT_SOLVER::calc_nb_points_col_matrix in matrix_v, unable to find ivnb: " << ivnb << " for iv: " << iv << ", points.boundary_flag[iv]: " << points.boundary_flag[iv] << "\n\n";
                    throw bad_exception();
                }
                nb_points_col_matrix_v[i1] = index;
            }
        }
}

void SEMI_IMPLICIT_SPLIT_SOLVER::single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, int it1, vector<int> &n_outer_iter_log, vector<double> &iterative_l1_err_log, vector<double> &iterative_max_err_log)
{
    it = it1;
    if (p_source.size() != p_new.size())
        p_new = Eigen::VectorXd::Zero(p_source.size());
    if (p_source.size() != p_old.size())
        p_old = Eigen::VectorXd::Zero(p_source.size());
    double total_steady_err;
    u_new = u_old, v_new = v_old, p_new = p_old; //initialize
    if (it % precond_freq_it == 0 || it == 0 || it == 1)
    {
        set_vel_matrix(points, cloud, parameters, u_new, v_new);
        solver_eigen_u.setTolerance(parameters.solver_tolerance); //default is machine precision (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#ac160a444af8998f93da9aa30e858470d)
        solver_eigen_u.setMaxIterations(parameters.n_iter);       //default is twice number of columns (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#af83de7a7d31d9d4bd1fef6222b07335b)
        solver_eigen_u.preconditioner().setDroptol(parameters.precond_droptol);
        solver_eigen_u.compute(matrix_u);
        solver_eigen_v.setTolerance(parameters.solver_tolerance); //default is machine precision (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#ac160a444af8998f93da9aa30e858470d)
        solver_eigen_v.setMaxIterations(parameters.n_iter);       //default is twice number of columns (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#af83de7a7d31d9d4bd1fef6222b07335b)
        solver_eigen_v.preconditioner().setDroptol(parameters.precond_droptol);
        solver_eigen_v.compute(matrix_v);
        calc_nb_points_col_matrix(points, cloud, parameters);
    }
    for (outer_iter = 0; outer_iter < n_outer_iter; outer_iter++)
    {
        u_iter_old = u_new, v_iter_old = v_new;
        modify_vel_matrix(points, cloud, parameters, u_new, v_new);
        calc_vel(points, parameters, u_new, v_new, p_new, u_old, v_old, p_old);
        calc_pressure(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old);
        calc_vel_corr(points, cloud, parameters, u_new, v_new);
        iterative_l1_err = (u_new - u_iter_old).lpNorm<1>() / (parameters.dimension * u_new.size());
        iterative_l1_err += ((v_new - v_iter_old).lpNorm<1>() / (parameters.dimension * v_new.size()));
        iterative_max_err = (u_new - u_iter_old).lpNorm<Eigen::Infinity>() / parameters.dimension;
        iterative_max_err += ((v_new - v_iter_old).lpNorm<Eigen::Infinity>() / parameters.dimension);
        if ((outer_iter == n_outer_iter - 1) || (iterative_l1_err <= iterative_tolerance))
        {
            iterative_max_err_log.push_back(iterative_max_err);
            iterative_l1_err_log.push_back(iterative_l1_err);
            n_outer_iter_log.push_back(outer_iter + 1);
            // printf("    SEMI_IMPLICIT_SPLIT_SOLVER::single_timestep_2d outer_iter: %i, iterative l1_err: %g, max_err: %g, tolerance: %g\n", outer_iter, iterative_l1_err, iterative_max_err, iterative_tolerance);
        }
        if (iterative_l1_err <= iterative_tolerance)
            break;
    }
    extras(points, parameters, u_new, v_new, p_new, u_old, v_old, p_old);
    // return total_steady_err;
}

void SEMI_IMPLICIT_SPLIT_SOLVER::single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y, int it1, vector<int> &n_outer_iter_log, vector<double> &iterative_l1_err_log, vector<double> &iterative_max_err_log)
{
    it = it1;
    if (p_source.size() != p_new.size())
        p_new = Eigen::VectorXd::Zero(p_source.size());
    if (p_source.size() != p_old.size())
        p_old = Eigen::VectorXd::Zero(p_source.size());
    double total_steady_err;
    u_new = u_old, v_new = v_old, p_new = p_old; //initialize
    if (it % precond_freq_it == 0 || it == 0 || it == 1)
    {
        set_vel_matrix(points, cloud, parameters, u_new, v_new);
        solver_eigen_u.setTolerance(parameters.solver_tolerance); //default is machine precision (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#ac160a444af8998f93da9aa30e858470d)
        solver_eigen_u.setMaxIterations(parameters.n_iter);       //default is twice number of columns (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#af83de7a7d31d9d4bd1fef6222b07335b)
        solver_eigen_u.preconditioner().setDroptol(parameters.precond_droptol);
        solver_eigen_u.compute(matrix_u);
        solver_eigen_v.setTolerance(parameters.solver_tolerance); //default is machine precision (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#ac160a444af8998f93da9aa30e858470d)
        solver_eigen_v.setMaxIterations(parameters.n_iter);       //default is twice number of columns (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#af83de7a7d31d9d4bd1fef6222b07335b)
        solver_eigen_v.preconditioner().setDroptol(parameters.precond_droptol);
        solver_eigen_v.compute(matrix_v);
        calc_nb_points_col_matrix(points, cloud, parameters);
    }
    for (outer_iter = 0; outer_iter < n_outer_iter; outer_iter++)
    {
        u_iter_old = u_new, v_iter_old = v_new;
        modify_vel_matrix(points, cloud, parameters, u_new, v_new);
        calc_vel(points, parameters, u_new, v_new, p_new, u_old, v_old, p_old, body_force_x, body_force_y);
        calc_pressure(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old);
        calc_vel_corr(points, cloud, parameters, u_new, v_new);
        iterative_l1_err = (u_new - u_iter_old).lpNorm<1>() / (parameters.dimension * u_new.size());
        iterative_l1_err += ((v_new - v_iter_old).lpNorm<1>() / (parameters.dimension * v_new.size()));
        iterative_max_err = (u_new - u_iter_old).lpNorm<Eigen::Infinity>() / parameters.dimension;
        iterative_max_err += ((v_new - v_iter_old).lpNorm<Eigen::Infinity>() / parameters.dimension);
        if ((outer_iter == n_outer_iter - 1) || (iterative_l1_err <= iterative_tolerance))
        {
            iterative_max_err_log.push_back(iterative_max_err);
            iterative_l1_err_log.push_back(iterative_l1_err);
            n_outer_iter_log.push_back(outer_iter + 1);
            // printf("    SEMI_IMPLICIT_SPLIT_SOLVER::single_timestep_2d outer_iter: %i, iterative l1_err: %g, max_err: %g, tolerance: %g\n", outer_iter, iterative_l1_err, iterative_max_err, iterative_tolerance);
        }
        if (iterative_l1_err <= iterative_tolerance)
            break;
    }
    extras(points, parameters, u_new, v_new, p_new, u_old, v_old, p_old);
    // return total_steady_err;
}