//Author: Dr. Shantanu Shahane
//compile: time make euler_steady_heat_cond_manuf_sol
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();

    string meshfile = "/media/shantanu/Data/All Simulation Results/Meshless_Methods/CAD_mesh_files/Square/gmsh/Square_n_40_unstruc.msh";
    double alpha = 0.1, max_err, l1_err, x, y;

    PARAMETERS parameters("parameters_file.csv", meshfile);
    POINTS points(parameters);
    CLOUD cloud(points, parameters);
    Eigen::VectorXd T_new = Eigen::VectorXd::Zero(points.nv), T_old = Eigen::VectorXd::Zero(points.nv), T_ana = Eigen::VectorXd::Zero(points.nv), T_source = Eigen::VectorXd::Zero(points.nv);
    vector<bool> dirichlet_flag;
    for (int iv = 0; iv < points.nv; iv++)
        dirichlet_flag.push_back(true);
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, 0.0, 0.0, 0.0, alpha);
    int dim = parameters.dimension;
    for (int iv = 0; iv < points.nv; iv++)
    {
        x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1];
        T_ana[iv] = sin(x) * sin(y);
        T_source[iv] = dim * alpha * T_ana[iv];
    }
    clock_t clock_t1 = clock(), clock_t2 = clock();
    for (int it = 0; it < parameters.nt; it++)
    {
        T_new = T_old + ((alpha * parameters.dt) * (points.laplacian_matrix_EIGEN_internal * T_old)) + (parameters.dt * T_source);
        for (int iv = 0; iv < points.nv; iv++)
            if (points.boundary_flag[iv])
                T_new[iv] = T_ana[iv]; //dirichlet BC

        calc_max_l1_error(T_old, T_new, max_err, l1_err);
        l1_err = l1_err / parameters.dt, max_err = max_err / parameters.dt;
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || l1_err < parameters.steady_tolerance)
        {
            printf("    main total steady state error: %g, steady_tolerance: %g\n", l1_err, parameters.steady_tolerance);
            printf("    main Completed it: %i of nt: %i, dt: %g, in CPU time: %.2g seconds\n\n", it, parameters.nt, parameters.dt, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
        if (l1_err < parameters.steady_tolerance && it > 1)
            break;
        T_old = T_new;
    }
    calc_max_l1_error(T_ana, T_new, max_err, l1_err);
    printf("\nmain dimension: %i, errors in T: max: %g, avg: %g\n", parameters.dimension, max_err, l1_err);
}