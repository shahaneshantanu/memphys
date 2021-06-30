//Author: Dr. Shantanu Shahane
//compile: time make heat_conduction_periodic
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();

    string meshfile = "../../gmsh_files/Square/Square_n_50_unstruc.msh"; //2D example
    // string meshfile = "../../gmsh_files/cuboid/Cuboid_n_20_unstruc.msh"; //3D example

    PARAMETERS parameters("parameters_file.csv", meshfile);
    POINTS points(parameters);
    vector<string> periodic_axis{"y"};
    points.set_periodic_bc(parameters, periodic_axis);
    CLOUD cloud(points, parameters);
    Eigen::VectorXd T_num_new = Eigen::VectorXd::Zero(points.nv);
    Eigen::VectorXd T_num_old = T_num_new, source = T_num_new;
    vector<bool> dirichlet_flag;

    double x, y, bc_mu = 0.5, bc_sig = 0.1;
    int dim = parameters.dimension;
    for (int iv = 0; iv < points.nv; iv++)
    {
        dirichlet_flag.push_back(true); //default is dirichlet
        x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1];
        if (points.boundary_flag[iv])
            if (fabs(x - 1.0) < 1E-4)
                T_num_old[iv] = exp(-(y - bc_mu) * (y - bc_mu) / (2 * bc_sig * bc_sig)) / (bc_sig * sqrt(2 * M_PI));
    }

    double max_err, l1_err;
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, 0.0, 0.0, 0.0, 1.0);
    clock_t t1 = clock();
    double unsteady_term_coeff = 1.0 / parameters.dt, conv_term_coeff = 0.0, diff_term_coeff = -1.0;
    SOLVER solver_T;
    solver_T.init(points, cloud, parameters, dirichlet_flag, unsteady_term_coeff, conv_term_coeff, diff_term_coeff, true);
    parameters.factoring_timer = ((double)(clock() - t1)) / CLOCKS_PER_SEC;
    clock_t clock_t1 = clock(), clock_t2 = clock();
    for (int it = 0; it < parameters.nt; it++)
    {
        source = T_num_old * unsteady_term_coeff;
        for (int iv = 0; iv < points.nv; iv++)
            if (points.boundary_flag[iv])
            {
                if (dirichlet_flag[iv])
                    source[iv] = T_num_old[iv];
                else
                    source[iv] = 0.0;
            }
        solver_T.general_solve(points, parameters, T_num_new, T_num_old, source);
        calc_max_l1_error(T_num_old, T_num_new, max_err, l1_err);
        l1_err = l1_err / parameters.dt, max_err = max_err / parameters.dt;
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || l1_err < parameters.steady_tolerance)
        {
            printf("    main total steady state error: %g, steady_tolerance: %g\n", l1_err, parameters.steady_tolerance);
            printf("    main Completed it: %i of nt: %i, dt: %g, in CPU time: %.2g seconds\n\n", it, parameters.nt, parameters.dt, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
        parameters.steady_error_log.push_back(l1_err);
        if (l1_err < parameters.steady_tolerance && it > 1)
            break;
        T_num_old = T_num_new;
    }
    parameters.solve_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    parameters.total_timer = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    write_simulation_details(points, cloud, parameters), write_iteration_details(parameters);
    vector<string> variable_names{"T_new"};
    vector<Eigen::VectorXd *> variable_pointers{&T_num_new};
    write_tecplot_steady_variables(points, parameters, variable_names, variable_pointers);
}