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

PARAMETERS::PARAMETERS(string parameter_file, string gmsh_file)
{
    cout << "\n";
    check_mpi();
    meshfile = gmsh_file;
    cout << "PARAMETERS::PARAMETERS gmsh_file: " << meshfile << endl;
    does_file_exist(meshfile.c_str(), "Called from PARAMETERS::read_calc_parameters");
    read_calc_parameters(parameter_file);
    verify_parameters();
    get_problem_dimension_msh();
    calc_cloud_num_points();
    calc_polynomial_term_exponents();
    cout << "\n";
}

void PARAMETERS::read_calc_parameters(string parameter_file)
{
    does_file_exist(parameter_file.c_str(), "Called from PARAMETERS::read_calc_parameters");
    char ctemp[5000], ctemp_2[100];
    string stemp;
    int itemp;
    double dtemp;
    FILE *file;
    file = fopen(parameter_file.c_str(), "r");
    cout << endl;
    fscanf(file, "%[^,],%i\n", ctemp, &poly_deg);
    printf("PARAMETERS::read_calc_parameters Read %s = %i\n", ctemp, poly_deg);
    fscanf(file, "%[^,],%i\n", ctemp, &phs_deg);
    printf("PARAMETERS::read_calc_parameters Read %s = %i\n", ctemp, phs_deg);
    fscanf(file, "%[^,],%lf\n", ctemp, &cloud_size_multiplier);
    printf("PARAMETERS::read_calc_parameters Read %s = %g\n", ctemp, cloud_size_multiplier);

    fscanf(file, "%[^,],%i\n", ctemp, &nt);
    printf("PARAMETERS::read_calc_parameters Read %s = %i\n", ctemp, nt);
    fscanf(file, "%[^,],%lf\n", ctemp, &Courant);
    printf("PARAMETERS::read_calc_parameters Read %s = %g\n", ctemp, Courant);
    fscanf(file, "%[^,],%s\n", ctemp, ctemp_2);
    solver_type = ctemp_2;
    printf("PARAMETERS::read_calc_parameters Read %s = %s\n", ctemp, solver_type.c_str());
    fscanf(file, "%[^,],%lf\n", ctemp, &steady_tolerance);
    printf("PARAMETERS::read_calc_parameters Read %s = %g\n", ctemp, steady_tolerance);
    fscanf(file, "%[^,],%lf\n", ctemp, &solver_tolerance);
    printf("PARAMETERS::read_calc_parameters Read %s = %g\n", ctemp, solver_tolerance);
    fscanf(file, "%[^,],%i\n", ctemp, &euclid_precond_level_hypre);
    printf("PARAMETERS::read_calc_parameters Read %s = %i\n", ctemp, euclid_precond_level_hypre);
    fscanf(file, "%[^,],%i\n", ctemp, &gmres_kdim);
    printf("PARAMETERS::read_calc_parameters Read %s = %i\n", ctemp, gmres_kdim);
    fscanf(file, "%[^,],%lf\n", ctemp, &precond_droptol);
    printf("PARAMETERS::read_calc_parameters Read %s = %g\n", ctemp, precond_droptol);
    fscanf(file, "%[^,],%i\n", ctemp, &n_iter);
    printf("PARAMETERS::read_calc_parameters Read %s = %i\n", ctemp, n_iter);

    cout << endl;
}

void PARAMETERS::verify_parameters()
{
    if (poly_deg < 2 || poly_deg > 15)
    {
        printf("\n\nERROR from PARAMETERS::verify_parameters poly_deg should be in range [2, 15]; current value: %i\n\n", poly_deg);
        throw bad_exception();
    }
    if (phs_deg != 3 && phs_deg != 5 && phs_deg != 7 && phs_deg != 9 && phs_deg != 11)
    {
        printf("\n\nERROR from PARAMETERS::verify_parameters phs_deg should be 3, 5, 7, 9, or 11; current value: %i\n\n", phs_deg);
        throw bad_exception();
    }
    if (strcmp(solver_type.c_str(), "hypre_ilu_gmres") != 0 && strcmp(solver_type.c_str(), "eigen_direct") != 0 && strcmp(solver_type.c_str(), "eigen_ilu_bicgstab") != 0)
    {
        cout << "\n\nERROR from PARAMETERS::verify_parameters solver_type should be either hypre_ilu_gmres, eigen_ilu_bicgstab or eigen_direct; current value: " << solver_type << "\n\n";
        throw bad_exception();
    }

    if (meshfile.find_last_of(".") + 1 == meshfile.size())
    {
        cout << "\n\nERROR from PARAMETERS::verify_parameters extension of meshfile should be msh; \nUnable to find extension in the meshfile: " << meshfile << "\n\n";
        throw bad_exception();
    }
    string extension = meshfile.substr(meshfile.find_last_of(".") + 1, meshfile.size());
    if (extension != "msh")
    {
        cout << "\n\nERROR from PARAMETERS::verify_parameters extension of meshfile should be msh; \nCurrent extension: " << extension << " of the meshfile: " << meshfile << "\n\n";
        throw bad_exception();
    }

    output_file_prefix = meshfile;
    if (output_file_prefix.find_last_of("/") < output_file_prefix.size()) //remove location details from mesh file name if required
        output_file_prefix = output_file_prefix.substr(output_file_prefix.find_last_of("/") + 1, output_file_prefix.size());
    if (output_file_prefix.find_last_of(".") < output_file_prefix.size()) //remove extension from mesh file name if required
        output_file_prefix.replace(output_file_prefix.begin() + output_file_prefix.find_last_of("."), output_file_prefix.end(), "");
    output_file_prefix += "_polydeg_" + to_string(poly_deg);
    cout << "PARAMETERS::verify_parameters output_file_prefix: " << output_file_prefix << endl;
}

void PARAMETERS::calc_cloud_num_points()
{
    if (dimension == 2)
        num_poly_terms = ((int)(0.5 * (poly_deg + 1) * (poly_deg + 2)));
    else
        num_poly_terms = ((int)((poly_deg + 1) * (poly_deg + 2) * (poly_deg + 3) / 6)); //sum(0.5*(1:poly_deg+1).*(2:poly_deg+2)) = (poly_deg + 1) * (poly_deg + 2) * (poly_deg + 3) / 6
    cloud_size = (int)(ceil(cloud_size_multiplier * num_poly_terms));
    cout << "PARAMETERS::calc_cloud_num_points num_poly_terms: " << num_poly_terms << ", cloud_size: " << cloud_size << endl;
}

void PARAMETERS::get_problem_dimension_msh()
{ //identify whether its a 2D or 3D gmsh grid

    clock_t start = clock();
    int dim = 2;
    FILE *file;
    int itemp, ncv, cv_type, eof_flag = 0;
    double dtemp;
    char temp[50];
    file = fopen(meshfile.c_str(), "r");
    while (true)
    {
        eof_flag = fscanf(file, "%s ", temp);
        if (strcmp(temp, "$MeshFormat") == 0)
            break;
        if (eof_flag < 0)
        {
            printf("\n\nERROR from PARAMETERS::get_problem_dimension_msh $MeshFormat not found in %s file\n\n", meshfile.c_str());
            throw bad_exception();
        }
    }
    // fscanf(file, "%s", temp);
    fgets(temp, 50, file);
    if (strcmp(temp, "2.2 0 8\n") != 0)
    {
        printf("\n\nERROR from PARAMETERS::get_problem_dimension_msh MeshFormat should be 2.2 0 8; but %s file has %s format instead\n\n", meshfile.c_str(), temp);
        throw bad_exception();
    }
    eof_flag = 0;
    while (true)
    {
        eof_flag = fscanf(file, "%s ", temp);
        if (strcmp(temp, "$Nodes") == 0)
            break;
        if (eof_flag < 0)
        {
            printf("\n\nERROR from PARAMETERS::get_problem_dimension_msh $Nodes not found in %s file\n\n", meshfile.c_str());
            throw bad_exception();
        }
    }
    eof_flag = 0;
    while (true)
    {
        eof_flag = fscanf(file, "%s ", temp);
        if (strcmp(temp, "$Elements") == 0)
            break;
        if (eof_flag < 0)
        {
            printf("\n\nERROR from PARAMETERS::get_problem_dimension_msh $Elements not found in %s file\n\n", meshfile.c_str());
            throw bad_exception();
        }
    }
    fscanf(file, "%i ", &ncv);
    // cout << "get_problem_dimension_msh ncv = " << ncv << endl;
    //reference: http://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_63.php
    for (int icv = 0; icv < ncv; icv++)
    {
        fscanf(file, "%i ", &itemp);   //cv number
        fscanf(file, "%i ", &cv_type); //cv type
        if (cv_type == 4 || cv_type == 5 || cv_type == 6 || cv_type == 7)
        { //4-node tetrahedron || 8-node hexahedron || 6-node prism || 5-node pyramid
            dim = 3;
            break; //3D element found
        }
        else if (cv_type == 11 || cv_type == 12 || cv_type == 13 || cv_type == 14)
        { //10-node tetrahedron || 27-node hexahedron || 18-node prism || 14-node pyramid
            dim = 3;
            break; //3D element found
        }
        else if (cv_type == 17 || cv_type == 18 || cv_type == 19)
        { //20-node hexahedron || 15-node prism || 13-node pyramid
            dim = 3;
            break; //3D element found
        }
        else if (cv_type == 29 || cv_type == 30 || cv_type == 31)
        { //20-node tetrahedron || 35-node tetrahedron || 56-node tetrahedron
            dim = 3;
            break; //3D element found
        }
        fscanf(file, "%*[^\n]\n"); //skip reading remaining row
    }
    dimension = dim;
    cout << "PARAMETERS::get_problem_dimension_msh problem dimension: " << dimension << endl;
}

void PARAMETERS::calc_polynomial_term_exponents()
{
    polynomial_term_exponents.resize(num_poly_terms, dimension);
    Eigen::MatrixXi deg_temp;
    if (dimension == 2)
    {
        polynomial_term_exponents.row(0) = Eigen::MatrixXi::Zero(1, dimension);                                        //constant term
        polynomial_term_exponents.block(1, 0, dimension, dimension) = Eigen::MatrixXi::Identity(dimension, dimension); //two linear terms
        int previous_index = 1;
        for (int deg = 2; deg <= poly_deg; deg++)
        { //quadratic onwards terms
            deg_temp.resize(deg + 1, dimension);
            for (int i = previous_index; i < previous_index + deg; i++)
            {
                deg_temp(i - previous_index, 0) = 1 + polynomial_term_exponents(i, 0); //exponents of 'x'
                deg_temp(i - previous_index, 1) = polynomial_term_exponents(i, 1);     //exponents of 'y'
            }
            deg_temp(deg, 0) = 0;
            deg_temp(deg, 1) = deg;
            previous_index = previous_index + deg;
            polynomial_term_exponents.block(previous_index, 0, deg + 1, dimension) = deg_temp;
        }
    }
    else
    {
        polynomial_term_exponents.row(0) = Eigen::MatrixXi::Zero(1, dimension);                                        //constant term
        polynomial_term_exponents.block(1, 0, dimension, dimension) = Eigen::MatrixXi::Identity(dimension, dimension); //three linear terms
        int previous_index = 1;
        for (int deg = 2; deg <= poly_deg; deg++)
        { //quadratic onwards terms
            deg_temp.resize(0.5 * (deg + 1) * (deg + 2), dimension);
            for (int i = previous_index; i < previous_index + (deg * (deg + 1) / 2); i++)
            {
                deg_temp(i - previous_index, 0) = 1 + polynomial_term_exponents(i, 0); //exponents of 'x'
                deg_temp(i - previous_index, 1) = polynomial_term_exponents(i, 1);     //exponents of 'y'
                deg_temp(i - previous_index, 2) = polynomial_term_exponents(i, 2);     //exponents of 'z'
            }
            for (int i = previous_index + (0.5 * deg * (deg - 1)); i < previous_index + (0.5 * deg * (deg + 1)); i++)
            {
                deg_temp(i - previous_index + deg, 0) = polynomial_term_exponents(i, 0);     //exponents of 'x'
                deg_temp(i - previous_index + deg, 1) = 1 + polynomial_term_exponents(i, 1); //exponents of 'y'
                deg_temp(i - previous_index + deg, 2) = polynomial_term_exponents(i, 2);     //exponents of 'z'
            }
            deg_temp((0.5 * (deg + 1) * (deg + 2)) - 1, 0) = 0;
            deg_temp((0.5 * (deg + 1) * (deg + 2)) - 1, 1) = 0;
            deg_temp((0.5 * (deg + 1) * (deg + 2)) - 1, 2) = deg;
            previous_index = previous_index + (0.5 * deg * (deg + 1));
            polynomial_term_exponents.block(previous_index, 0, 0.5 * (deg + 1) * (deg + 2), dimension) = deg_temp;
        }
    }
}

void PARAMETERS::calc_dt(Eigen::SparseMatrix<double, Eigen::RowMajor> &grad_x, Eigen::SparseMatrix<double, Eigen::RowMajor> &grad_y, Eigen::SparseMatrix<double, Eigen::RowMajor> &grad_z, Eigen::SparseMatrix<double, Eigen::RowMajor> &laplacian, double u0, double v0, double w0, double alpha)
{
    Eigen::VectorXcd eigval_grad_x, eigval_grad_y, eigval_grad_z, eigval_laplacian;
    eigval_grad_x = calc_largest_magnitude_eigenvalue(grad_x);
    eigval_grad_y = calc_largest_magnitude_eigenvalue(grad_y);
    if (dimension == 3)
        eigval_grad_z = calc_largest_magnitude_eigenvalue(grad_z);
    eigval_laplacian = calc_largest_magnitude_eigenvalue(laplacian);
    grad_x_eigval_real = eigval_grad_x[0].real(), grad_x_eigval_imag = eigval_grad_x[0].imag();
    grad_y_eigval_real = eigval_grad_y[0].real(), grad_y_eigval_imag = eigval_grad_y[0].imag();
    if (dimension == 3)
        grad_z_eigval_real = eigval_grad_z[0].real(), grad_z_eigval_imag = eigval_grad_z[0].imag();
    laplace_eigval_real = eigval_laplacian[0].real(), laplace_eigval_imag = eigval_laplacian[0].imag();
    printf("\nPARAMETERS::calc_dt Eigenvalue with largest magnitude: grad_x: (%g, %g)\n", eigval_grad_x[0].real(), eigval_grad_x[0].imag());
    printf("PARAMETERS::calc_dt Eigenvalue with largest magnitude: grad_y: (%g, %g)\n", eigval_grad_y[0].real(), eigval_grad_y[0].imag());
    if (dimension == 3)
        printf("PARAMETERS::calc_dt Eigenvalue with largest magnitude: grad_z: (%g, %g)\n", eigval_grad_z[0].real(), eigval_grad_z[0].imag());
    printf("PARAMETERS::calc_dt Eigenvalue with largest magnitude: laplacian: (%g, %g)\n", eigval_laplacian[0].real(), eigval_laplacian[0].imag());

    double evalues = u0 * sqrt((eigval_grad_x[0].real() * eigval_grad_x[0].real()) + (eigval_grad_x[0].imag() * eigval_grad_x[0].imag()));
    evalues += v0 * sqrt((eigval_grad_y[0].real() * eigval_grad_y[0].real()) + (eigval_grad_y[0].imag() * eigval_grad_y[0].imag()));
    if (dimension == 3)
        evalues += w0 * sqrt((eigval_grad_z[0].real() * eigval_grad_z[0].real()) + (eigval_grad_z[0].imag() * eigval_grad_z[0].imag()));
    evalues += alpha * sqrt((eigval_laplacian[0].real() * eigval_laplacian[0].real()) + (eigval_laplacian[0].imag() * eigval_laplacian[0].imag()));
    dt = 2.0 / evalues; //forward Euler
    dt = dt * Courant;
    printf("PARAMETERS::calc_dt Courant: %g, dt: %g seconds\n\n", Courant, dt);
}