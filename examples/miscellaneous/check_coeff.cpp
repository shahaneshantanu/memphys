//Author: Dr. Shantanu Shahane
//compile: time make check_coeff
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
#include "../../header_files/coefficient_computations.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();

    string meshfile = "/media/shantanu/Data/All Simulation Results/Meshless_Methods/CAD_mesh_files/Square/gmsh/Square_n_10_unstruc.msh"; //2D example
    // string meshfile = "/media/shantanu/Data/All Simulation Results/Meshless_Methods/CAD_mesh_files/cuboid/Cuboid_n_10_unstruc.msh"; //3D example

    PARAMETERS parameters("parameters_file.csv", meshfile);
    POINTS points(parameters);
    CLOUD cloud(points, parameters);
    int dim = parameters.dimension;
    write_csv(points.xyz, points.nv, dim, "xyz.csv");

    // print_to_terminal(parameters.polynomial_term_exponents, "main parameters.polynomial_term_exponents");

    int iv_check = 429, ivnb;
    if (dim == 2)
        printf("\niv_check: %i, x: %g, y: %g\n", iv_check, points.xyz[dim * iv_check], points.xyz[dim * iv_check + 1]);
    else
        printf("\niv_check: %i, x: %g, y: %g, z: %g\n", iv_check, points.xyz[dim * iv_check], points.xyz[dim * iv_check + 1], points.xyz[dim * iv_check + 2]);
    cout << "\nCloud points of iv_check:\n";
    for (int i1 = cloud.nb_points_row[iv_check]; i1 < cloud.nb_points_row[iv_check + 1]; i1++)
    {
        ivnb = cloud.nb_points_col[i1];
        if (dim == 2)
            printf("ivnb: %i, x: %g, y: %g\n", ivnb, points.xyz[dim * ivnb], points.xyz[dim * ivnb + 1]);
        else
            printf("ivnb: %i, x: %g, y: %g, z: %g\n", ivnb, points.xyz[dim * ivnb], points.xyz[dim * ivnb + 1], points.xyz[dim * ivnb + 2]);
    }
    cout << "Cloud points of iv_check:\n\n";
    Eigen::MatrixXd grad_laplace;
    grad_laplace.resize(parameters.cloud_size, dim + 1);
    for (int i1 = cloud.nb_points_row[iv_check]; i1 < cloud.nb_points_row[iv_check + 1]; i1++)
    {
        grad_laplace(i1 - cloud.nb_points_row[iv_check], 0) = cloud.grad_x_coeff[i1];
        grad_laplace(i1 - cloud.nb_points_row[iv_check], 1) = cloud.grad_y_coeff[i1];
        if (dim == 3)
            grad_laplace(i1 - cloud.nb_points_row[iv_check], 2) = cloud.grad_z_coeff[i1];
        grad_laplace(i1 - cloud.nb_points_row[iv_check], dim) = cloud.laplacian_coeff[i1];
    }
    print_to_terminal(grad_laplace, "main grad_laplace");
}