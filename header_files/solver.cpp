//Author: Dr. Shantanu Shahane
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include "class.hpp"
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
using namespace std;

void SOLVER::init(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &dirichlet_flag, double unsteady_term_coeff, double conv_term_coeff, double diff_term_coeff, bool log_flag)
{
    solver_tolerance = parameters.solver_tolerance;
    solver_type = parameters.solver_type;
    euclid_precond_level_hypre = parameters.euclid_precond_level_hypre;
    precond_droptol = parameters.precond_droptol;
    gmres_kdim = parameters.gmres_kdim;
    precond_droptol = parameters.precond_droptol;
    n_iter = parameters.n_iter;
    dirichlet_flag_1 = dirichlet_flag;
    unsteady_term_coeff_1 = unsteady_term_coeff;
    conv_term_coeff_1 = conv_term_coeff;
    diff_term_coeff_1 = diff_term_coeff;
    log_flag_1 = log_flag;
    system_size = points.nv + 1; //initialize assuming full Neumann [extra constraint (regularization: http://www-e6.ijs.si/medusa/wiki/index.php/Poisson%27s_equation) for neumann BCs]
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv] && dirichlet_flag_1[iv])
        { //identified at least one boundary point with dirichlet BC (no regularization needed)
            system_size = points.nv;
            break;
        }

    calc_coeff_matrix(points, cloud, parameters);
    if (strcmp(solver_type.c_str(), "hypre_ilu_gmres") == 0)
    {
        scale_coeff(points);
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, system_size - 1, 0, system_size - 1, &coeff_HYPRE);
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, system_size - 1, &source_HYPRE);
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, system_size - 1, &X_HYPRE);
        HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver_X_HYPRE);
        HYPRE_EuclidCreate(MPI_COMM_WORLD, &precond_X_HYPRE);
        set_solve_parameters();
        HYPRE_set_coeff(parameters);
        HYPRE_GMRESSetPrecond(solver_X_HYPRE, (HYPRE_PtrToSolverFcn)HYPRE_EuclidSolve, (HYPRE_PtrToSolverFcn)HYPRE_EuclidSetup, precond_X_HYPRE);
        HYPRE_ParCSRGMRESSetup(solver_X_HYPRE, parcsr_coeff_HYPRE, par_source_HYPRE, par_X_HYPRE);
        // HYPRE_IJMatrixPrint(coeff_HYPRE, "coeff_hypre_T");
    }
    else if (strcmp(solver_type.c_str(), "eigen_direct") == 0)
    {
        EIGEN_set_coeff();
        solver_eigen_direct.analyzePattern(coeff_EIGEN);
        solver_eigen_direct.factorize(coeff_EIGEN);
    }
    else if (strcmp(solver_type.c_str(), "eigen_ilu_bicgstab") == 0)
    {
        EIGEN_set_coeff();
        set_solve_parameters();
        // solver_eigen_ilu_bicgstab.analyzePattern(coeff_EIGEN);
        solver_eigen_ilu_bicgstab.compute(coeff_EIGEN);
    }
    else
    {
        cout << "\n\nERROR from SOLVER::SOLVER solver_type should be either hypre_ilu_gmres, eigen_ilu_bicgstab or eigen_direct; current value: " << solver_type << "\n\n";
        throw bad_exception();
    }
}

void SOLVER::EIGEN_set_coeff()
{
    int nnz = 1, iv_1, iv_2;
    double val;
    vector<Eigen::Triplet<double>> triplet;
    for (int i1 = 0; i1 < coeff_matrix.size(); i1++)
    {
        iv_1 = get<0>(coeff_matrix[i1]);
        iv_2 = get<1>(coeff_matrix[i1]);
        val = get<2>(coeff_matrix[i1]);
        triplet.push_back(Eigen::Triplet<double>(iv_1, iv_2, val));
    }
    coeff_EIGEN.resize(system_size, system_size);
    coeff_EIGEN.setFromTriplets(triplet.begin(), triplet.end());
    coeff_EIGEN.makeCompressed();
    triplet.clear();
}

void SOLVER::HYPRE_set_coeff(PARAMETERS &parameters)
{
    int nnz = 1, iv_1, iv_2, dim = parameters.dimension;
    double val;
    HYPRE_IJMatrixSetObjectType(coeff_HYPRE, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(coeff_HYPRE);
    for (int i1 = 0; i1 < coeff_matrix.size(); i1++)
    {
        iv_1 = get<0>(coeff_matrix[i1]);
        iv_2 = get<1>(coeff_matrix[i1]);
        val = get<2>(coeff_matrix[i1]);
        HYPRE_IJMatrixAddToValues(coeff_HYPRE, 1, &nnz, &iv_1, &iv_2, &val);
    }
    HYPRE_IJMatrixAssemble(coeff_HYPRE);
    HYPRE_IJMatrixGetObject(coeff_HYPRE, (void **)&parcsr_coeff_HYPRE);

    HYPRE_IJVectorSetObjectType(source_HYPRE, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(source_HYPRE);
    HYPRE_IJVectorSetObjectType(X_HYPRE, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(X_HYPRE);
    rows_HYPRE = new int[system_size];
    X = new double[system_size];
    source = new double[system_size];
    for (int iv = 0; iv < system_size; iv++)
    {
        X[iv] = 0.0;
        source[iv] = 0.0;
        rows_HYPRE[iv] = iv;
    }
    HYPRE_IJVectorSetValues(source_HYPRE, system_size, rows_HYPRE, source);
    HYPRE_IJVectorAssemble(source_HYPRE);
    HYPRE_IJVectorGetObject(source_HYPRE, (void **)&par_source_HYPRE);
    HYPRE_IJVectorSetValues(X_HYPRE, system_size, rows_HYPRE, X);
    HYPRE_IJVectorAssemble(X_HYPRE);
    HYPRE_IJVectorGetObject(X_HYPRE, (void **)&par_X_HYPRE);
}

void SOLVER::set_solve_parameters()
{
    if (strcmp(solver_type.c_str(), "hypre_ilu_gmres") == 0)
    {
        HYPRE_GMRESSetMaxIter(solver_X_HYPRE, n_iter);               /* max iterations */
        HYPRE_GMRESSetTol(solver_X_HYPRE, 0.0);                      /* conv. tolerance relative to norm(RHS source term)*/
        HYPRE_GMRESSetAbsoluteTol(solver_X_HYPRE, solver_tolerance); /* absolute conv. tolerance */
        HYPRE_GMRESSetPrintLevel(solver_X_HYPRE, print_flag);
        HYPRE_GMRESSetKDim(solver_X_HYPRE, gmres_kdim);
        // HYPRE_ParCSRPilutSetDropTolerance(precond_X_HYPRE, 1E-5);
        HYPRE_EuclidSetLevel(precond_X_HYPRE, euclid_precond_level_hypre);
        HYPRE_EuclidSetSparseA(precond_X_HYPRE, precond_droptol);
        HYPRE_EuclidSetRowScale(precond_X_HYPRE, 1);
        // HYPRE_EuclidSetMem(precond_X_HYPRE, 1);
        HYPRE_EuclidSetStats(precond_X_HYPRE, 1);
    }
    else if (strcmp(solver_type.c_str(), "eigen_ilu_bicgstab") == 0)
    {
        solver_eigen_ilu_bicgstab.setTolerance(solver_tolerance); //default is machine precision (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#ac160a444af8998f93da9aa30e858470d)
        solver_eigen_ilu_bicgstab.setMaxIterations(n_iter);       //default is twice number of columns (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#af83de7a7d31d9d4bd1fef6222b07335b)
        solver_eigen_ilu_bicgstab.preconditioner().setDroptol(precond_droptol);
    }
    else
    {
        cout << "\n\nERROR from SOLVER::SOLVER solver_type should be hypre_ilu_gmres or eigen_ilu_bicgstab; current value: " << solver_type << "\n\n";
        throw bad_exception();
    }
}

void SOLVER::calc_coeff_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters)
{
    int dim = parameters.dimension, iv_nb;
    double val;
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (points.boundary_flag[iv])
        { //lies on boundary
            if (dirichlet_flag_1[iv])
            { //apply dirichlet BC
                coeff_matrix.push_back(tuple<int, int, double>(iv, iv, 1.0));
            }
            else
            { //apply neumann BC
                for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                {
                    iv_nb = cloud.nb_points_col[i1];
                    val = points.normal[dim * iv] * cloud.grad_x_coeff[i1] + points.normal[dim * iv + 1] * cloud.grad_y_coeff[i1];
                    if (dim == 3)
                        val += points.normal[dim * iv + 2] * cloud.grad_z_coeff[i1];
                    coeff_matrix.push_back(tuple<int, int, double>(iv, iv_nb, val));
                }
            }
        }
        else
        {
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                iv_nb = cloud.nb_points_col[i1];
                val = diff_term_coeff_1 * cloud.laplacian_coeff[i1]; //diffusion
                val += conv_term_coeff_1 * cloud.grad_x_coeff[i1];   //convection
                val += conv_term_coeff_1 * cloud.grad_y_coeff[i1];   //convection
                if (dim == 3)
                    val += conv_term_coeff_1 * cloud.grad_z_coeff[i1]; //convection
                if (iv_nb == iv)
                    val += unsteady_term_coeff_1; //diagonal term
                coeff_matrix.push_back(tuple<int, int, double>(iv, iv_nb, val));
            }
        }
    }
    if (system_size == points.nv + 1)
    { //extra constraint (regularization: http://www-e6.ijs.si/medusa/wiki/index.php/Poisson%27s_equation) for neumann BCs
        for (int iv = 0; iv < points.nv; iv++)
        {
            coeff_matrix.push_back(tuple<int, int, double>(iv, points.nv, 1.0)); //last column
            coeff_matrix.push_back(tuple<int, int, double>(points.nv, iv, 1.0)); //last row
        }
        coeff_matrix.push_back(tuple<int, int, double>(points.nv, points.nv, 0.0)); //last entry
    }
}

void SOLVER::scale_coeff(POINTS &points)
{
    for (int i1 = 0; i1 < system_size; i1++)
        scale.push_back(1.0);
    int row, col;
    double dtemp;
    for (int i1 = 0; i1 < coeff_matrix.size(); i1++)
    { //calculate scale
        row = get<0>(coeff_matrix[i1]);
        if (row < points.nv)
        { //for neumann BC, last row diagonal is zero (to avoid division by zero)
            col = get<1>(coeff_matrix[i1]);
            if (row == col)
            { //diagonal entry of assembled matrix
                dtemp = get<2>(coeff_matrix[i1]);
                scale[row] = 1.0 / dtemp;
            }
        }
    }
    for (int i1 = 0; i1 < coeff_matrix.size(); i1++)
    { //scale coeff_matrix
        row = get<0>(coeff_matrix[i1]);
        col = get<1>(coeff_matrix[i1]);
        get<2>(coeff_matrix[i1]) = get<2>(coeff_matrix[i1]) * scale[row];
    }
}

void SOLVER::general_solve(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &field_new, Eigen::VectorXd &field_old, Eigen::VectorXd &rhs)
{
    double absolute_residual, relative_residual;
    int iter;
    if (strcmp(solver_type.c_str(), "hypre_ilu_gmres") == 0)
    {
        for (int iv = 0; iv < rhs.size(); iv++)
            source[iv] = rhs(iv) * scale[iv];
        HYPRE_IJVectorSetValues(source_HYPRE, system_size, rows_HYPRE, source);
        HYPRE_IJVectorAssemble(source_HYPRE);
        HYPRE_IJVectorGetObject(source_HYPRE, (void **)&par_source_HYPRE);
        for (int iv = 0; iv < field_old.size(); iv++)
            X[iv] = field_old[iv]; //initialize with last time step value
        HYPRE_IJVectorSetValues(X_HYPRE, system_size, rows_HYPRE, X);
        HYPRE_IJVectorAssemble(X_HYPRE);
        HYPRE_IJVectorGetObject(X_HYPRE, (void **)&par_X_HYPRE);

        HYPRE_ParCSRGMRESSolve(solver_X_HYPRE, parcsr_coeff_HYPRE, par_source_HYPRE, par_X_HYPRE);
        HYPRE_GMRESGetNumIterations(solver_X_HYPRE, &iter);
        HYPRE_GMRESGetFinalRelativeResidualNorm(solver_X_HYPRE, &relative_residual);
        l2_norm = vector_norm(source, system_size, 2);
        absolute_residual = relative_residual * l2_norm;

        HYPRE_IJVectorGetValues(X_HYPRE, system_size, rows_HYPRE, X);
        for (int iv = 0; iv < system_size; iv++)
            field_new[iv] = X[iv];
        if (log_flag_1)
        {
            parameters.rel_res_log.push_back(relative_residual);
            parameters.abs_res_log.push_back(absolute_residual);
            parameters.n_iter_actual.push_back(iter);
            if (system_size == points.nv + 1)
                parameters.regul_alpha_log.push_back(field_new[system_size - 1]);
        }
        if (absolute_residual > solver_tolerance && relative_residual > solver_tolerance)
        {
            printf("\n\nERROR from SOLVER::general_solve did not converge in %i iterations, relative residual: %g, absolute residual: %g, l2_norm(source): %g\n\n", iter, relative_residual, absolute_residual, l2_norm);
            throw bad_exception();
        }
    }
    else if (strcmp(solver_type.c_str(), "eigen_direct") == 0)
    {
        field_new = solver_eigen_direct.solve(rhs);
        if (log_flag_1)
        {
            absolute_residual = (coeff_EIGEN * field_new - rhs).norm();
            relative_residual = absolute_residual / rhs.norm();
            parameters.rel_res_log.push_back(relative_residual);
            parameters.abs_res_log.push_back(absolute_residual);
            parameters.n_iter_actual.push_back(1);
            if (system_size == points.nv + 1)
                parameters.regul_alpha_log.push_back(field_new[system_size - 1]);
        }
    }
    else if (strcmp(solver_type.c_str(), "eigen_ilu_bicgstab") == 0)
    {
        field_new = solver_eigen_ilu_bicgstab.solveWithGuess(rhs, field_old);
        if (log_flag_1)
        {
            absolute_residual = (coeff_EIGEN * field_new - rhs).norm();
            relative_residual = absolute_residual / rhs.norm();
            parameters.rel_res_log.push_back(relative_residual);
            parameters.abs_res_log.push_back(absolute_residual);
            parameters.n_iter_actual.push_back(solver_eigen_ilu_bicgstab.iterations());
            if (system_size == points.nv + 1)
                parameters.regul_alpha_log.push_back(field_new[system_size - 1]);
        }
    }
    else
    {
        cout << "\n\nERROR from SOLVER::SOLVER solver_type should be either hypre_ilu_gmres, eigen_ilu_bicgstab or eigen_direct; current value: " << solver_type << "\n\n";
        throw bad_exception();
    }
    if (system_size == points.nv + 1) //old regularization alpha reset to new
        field_old[system_size - 1] = field_new[system_size - 1];
}