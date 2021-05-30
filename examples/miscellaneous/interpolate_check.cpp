//Author: Dr. Shantanu Shahane
//compile: time make interpolate_check
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
    // string meshfile = "/media/shantanu/Datall Simulation Results/Meshless_Methods/CAD_mesh_files/cuboid/Cuboid_n_10_unstruc.msh"; //3D example

    PARAMETERS parameters("parameters_file.csv", meshfile);
    POINTS points(parameters);
    CLOUD cloud(points, parameters);
    int dim = parameters.dimension, nv_probe = dim * 6;
    vector<double> xyz_probe;
    for (int iv = 0; iv < nv_probe / dim; iv++)
    {
        xyz_probe.push_back(iv / ((nv_probe / dim) - 1.0));
        xyz_probe.push_back(0.5);
        if (dim == 3)
            xyz_probe.push_back(0.5);
    }
    for (int iv = 0; iv < nv_probe / dim; iv++)
    {
        xyz_probe.push_back(0.5);
        xyz_probe.push_back(iv / ((nv_probe / dim) - 1.0));
        if (dim == 3)
            xyz_probe.push_back(0.5);
    }
    if (dim == 3)
        for (int iv = 0; iv < nv_probe / dim; iv++)
        {
            xyz_probe.push_back(0.5);
            xyz_probe.push_back(0.5);
            xyz_probe.push_back(iv / ((nv_probe / dim) - 1.0));
        }

    Eigen::SparseMatrix<double, Eigen::RowMajor> interp_matrix = calc_interp_matrix(xyz_probe, points, parameters);

    Eigen::VectorXd field = Eigen::VectorXd::Zero(points.nv);
    Eigen::VectorXd interp_ana = Eigen::VectorXd::Zero(nv_probe), interp_num = interp_ana;
    if (dim == 2)
    {
        for (int iv = 0; iv < points.nv; iv++)
            field[iv] = exp(points.xyz[dim * iv] + points.xyz[dim * iv + 1]);
        for (int iv = 0; iv < nv_probe; iv++)
            interp_ana[iv] = exp(xyz_probe[dim * iv] + xyz_probe[dim * iv + 1]);
    }
    else
    {
        for (int iv = 0; iv < points.nv; iv++)
            field[iv] = exp(points.xyz[dim * iv] + points.xyz[dim * iv + 1] + points.xyz[dim * iv + 2]);
        for (int iv = 0; iv < nv_probe; iv++)
            interp_ana[iv] = exp(xyz_probe[dim * iv] + xyz_probe[dim * iv + 1] + xyz_probe[dim * iv + 2]);
    }
    interp_num = interp_matrix * field;
    double max_err, l1_err;
    calc_max_l1_error(interp_ana, interp_num, max_err, l1_err);
    printf("\n\nmain error in interp: max: %g, l1: %g\n\n", max_err, l1_err);
}