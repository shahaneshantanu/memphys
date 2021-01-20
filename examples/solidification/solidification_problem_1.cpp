//Author: Dr. Shantanu Shahane
//compile: time make solidification_2d_1
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    double T_init = 1000.0, T_wall = 700.0;
    int tecplot_it_frequency = 100, temporal_order = 2;
    // string meshfile = "/home/shantanu/Desktop/All Simulation Results/Meshless_Methods/CAD_mesh_files/solidification/Square/Square_n_20_unstruc.msh";
    string meshfile = "/home/shantanu/Desktop/All Simulation Results/Meshless_Methods/CAD_mesh_files/solidification/complex_2D/mesh_n_15_r_5.msh";
    // string meshfile = "/home/shantanu/Desktop/All Simulation Results/Meshless_Methods/CAD_mesh_files/solidification/shaft_holder/mesh_maxelem_3.msh";
    // string meshfile = "/home/shantanu/Desktop/All Simulation Results/Meshless_Methods/CAD_mesh_files/solidification/mixer_pipe/mesh_maxelem_8.msh"; //does not work: gives memory fault in function "re_order_points(points, parameters)": not sure about reason
    PARAMETERS parameters("parameters_file.csv", meshfile);
    int dim = parameters.dimension;
    parameters.Courant = parameters.Courant / ((double)temporal_order); //Adam-Bashforth has half stability than explicit Euler;
    POINTS points(parameters);
    if (dim == 3)
    {
        double scale = 0.001;
        for (int iv = 0; iv < dim * points.nv; iv++)
            points.xyz[iv] = scale * points.xyz[iv];
        for (int iv = 0; iv < dim * points.nv_original; iv++)
            points.xyz_original[iv] = scale * points.xyz_original[iv];
    }
    CLOUD cloud(points, parameters);
    Eigen::VectorXd T_new = Eigen::VectorXd::Zero(points.nv), fs_new = Eigen::VectorXd::Zero(points.nv);
    double x, y, z;
    for (int iv = 0; iv < points.nv; iv++)
    {
        x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1];
        if (dim == 3)
            z = points.xyz[dim * iv + 2];
        T_new[iv] = T_init, fs_new[iv] = 0.0;
        if (points.boundary_flag[iv])
            T_new[iv] = T_wall, fs_new[iv] = 1.0;
    }
    Eigen::VectorXd T_old = T_new, fs_old = fs_new;
    SOLIDIFICATION solidification(points, cloud, parameters, temporal_order);
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, 0.0, 0.0, 0.0, solidification.alpha);

    clock_t clock_t1 = clock(), clock_t2 = clock();
    double T_max = 0.0;
    solidification.write_tecplot_temporal_header(points, parameters);
    solidification.write_tecplot_temporal_fields(points, parameters, T_new, fs_new, 0);
    for (int it = 0; it < parameters.nt; it++)
    {
        solidification.single_timestep_2d(points, cloud, parameters, T_new, T_old, fs_new, fs_old, it);
        T_max = max_abs(T_new);
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || T_max < (solidification.Tsol - 5.0))
        {
            printf("    main T_max: %g, Tsol: %g\n", T_max, solidification.Tsol - 5.0);
            printf("    main Completed it: %i of nt: %i, dt: %g, in CPU time: %.2g seconds\n\n", it, parameters.nt, parameters.dt, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
        if (tecplot_it_frequency > 0 && it > 0)
            if (it % tecplot_it_frequency == 0 || T_max < (solidification.Tsol - 5.0) || it == parameters.nt - 1)
                solidification.write_tecplot_temporal_fields(points, parameters, T_new, fs_new, it);
        if (T_max < (solidification.Tsol - 5.0))
        {
            printf("\n\nTotal Solidification Time: %g seconds\n\n", ((double)it) * parameters.dt);
            break;
        }
        T_old = T_new, fs_old = fs_new;
    }
    parameters.solve_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    parameters.total_timer = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    write_simulation_details(points, cloud, parameters); //, write_iteration_details(parameters);
}