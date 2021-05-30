//Author: Dr. Shantanu Shahane
//compile: make main_scalar_transport_implicit
//execute: ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    double wv = 1.0; //wavenumber

    string meshfile = "/media/shantanu/Data/All Simulation Results/Meshless_Methods/CAD_mesh_files/Square/gmsh/Square_n_10_unstruc.msh"; //2D example
    // string meshfile = "/media/shantanu/Data/All Simulation Results/Meshless_Methods/CAD_mesh_files/cuboid/Cuboid_n_10_unstruc.msh"; //3D example

    PARAMETERS parameters("parameters_file.csv", meshfile);
    POINTS points(parameters);
    CLOUD cloud(points, parameters);
    Eigen::VectorXd T_ana = Eigen::VectorXd::Zero(points.nv), T_num_new = Eigen::VectorXd::Zero(points.nv), T_num_old = Eigen::VectorXd::Zero(points.nv), source = Eigen::VectorXd::Zero(points.nv);
    double max_err, l1_err;
    int dim = parameters.dimension;
    for (int iv = 0; iv < points.nv; iv++)
    { //manufactured solution: sin(wv*x)*sin(wv*y)*sin(wv*z)
        T_ana[iv] = sin(wv * points.xyz[dim * iv]) * sin(wv * points.xyz[dim * iv + 1]);
        if (dim == 3)
            T_ana[iv] = T_ana[iv] * sin(wv * points.xyz[dim * iv + 2]);
    }
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, 0.0, 0.0, 0.0, parameters.mu / parameters.rho);
    double unsteady_term_coeff = 1.0 / parameters.dt, conv_term_coeff = 0.1, diff_term_coeff = -1.0;
    vector<bool> dirichlet_flag;
    for (int iv = 0; iv < points.nv; iv++)
        dirichlet_flag.push_back(true);
    clock_t t1 = clock();
    SOLVER solver_T_dirichlet;
    solver_T_dirichlet.init(points, cloud, parameters, dirichlet_flag, unsteady_term_coeff, conv_term_coeff, diff_term_coeff, true);
    parameters.factoring_timer = ((double)(clock() - t1)) / CLOCKS_PER_SEC;
    double manuf_source_conv_x, manuf_source_conv_y, manuf_source_conv_z = 0.0;
    clock_t t2 = clock(), clock_t2 = clock();
    for (int it = 0; it < parameters.nt; it++)
    {
        for (int iv = 0; iv < points.nv; iv++)
        {
            if (points.boundary_flag[iv])
                source[iv] = T_ana[iv]; //dirichlet BC
            else
            {
                //manufactured solution source term: diffusion
                source[iv] = -diff_term_coeff * wv * wv * dim * T_ana[iv];
                //manufactured solution source term: convection
                manuf_source_conv_x = wv * cos(wv * points.xyz[dim * iv]) * sin(wv * points.xyz[dim * iv + 1]);
                manuf_source_conv_y = wv * sin(wv * points.xyz[dim * iv]) * cos(wv * points.xyz[dim * iv + 1]);
                if (dim == 3)
                {
                    manuf_source_conv_x = manuf_source_conv_x * sin(wv * points.xyz[dim * iv + 2]);
                    manuf_source_conv_y = manuf_source_conv_y * sin(wv * points.xyz[dim * iv + 2]);
                    manuf_source_conv_z = wv * sin(wv * points.xyz[dim * iv]) * sin(wv * points.xyz[dim * iv + 1]) * cos(wv * points.xyz[dim * iv + 2]);
                }
                source[iv] = source[iv] + conv_term_coeff * (manuf_source_conv_x + manuf_source_conv_y + manuf_source_conv_z);
                source[iv] = source[iv] + (T_num_old[iv] * unsteady_term_coeff); //unsteady term
            }
        }
        solver_T_dirichlet.general_solve(points, parameters, T_num_new, T_num_old, source);
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        calc_max_l1_relative_error(T_num_old, T_num_new, max_err, l1_err);
        l1_err = l1_err / parameters.dt, max_err = max_err / parameters.dt;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || l1_err < parameters.steady_tolerance)
            printf("    main dimension: %i, it: %i, steady state errors in T: max: %g, avg: %g\n", parameters.dimension, it, max_err, l1_err), clock_t2 = clock();
        parameters.steady_error_log.push_back(l1_err);
        if (l1_err < parameters.steady_tolerance && it > 1)
            break;
        T_num_old = T_num_new;
    }
    parameters.solve_timer = ((double)(clock() - t2)) / CLOCKS_PER_SEC;
    calc_max_l1_relative_error(T_ana, T_num_new, max_err, l1_err);
    printf("\nmain Dirichlet BC dimension: %i, relative errors in T: max: %g, avg: %g\n\n\n", dim, max_err, l1_err);
    parameters.total_timer = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    write_simulation_details(points, cloud, parameters), write_iteration_details(parameters);
    Eigen::VectorXd T_error = (T_ana - T_num_new).cwiseAbs();
    vector<string> variable_names{"T_new", "T_ana", "T_error"};
    vector<Eigen::VectorXd *> variable_pointers{&T_num_new, &T_ana, &T_error};
    write_tecplot_steady_variables(points, parameters, variable_names, variable_pointers);
}