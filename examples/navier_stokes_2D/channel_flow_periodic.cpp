//Author: Dr. Shantanu Shahane
//compile: time make channel_flow_periodic
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    PARAMETERS parameters("parameters_file.csv", "../../gmsh_files/Rectangle/Rectangle_n_10.msh");
    int dim = parameters.dimension, temporal_order = 2;
    double iterative_tolerance = 1E-5; //parameters.steady_tolerance;
    int precond_freq_it = 10000, n_outer_iter = 5;

    POINTS points(parameters);
    vector<string> periodic_axis{"x"};
    points.set_periodic_bc(parameters, periodic_axis);
    CLOUD cloud(points, parameters);

    Eigen::VectorXd p_new = Eigen::VectorXd::Zero(points.nv);
    Eigen::VectorXd u_new = Eigen::VectorXd::Zero(points.nv), v_new = u_new;
    Eigen::VectorXd u_old = u_new, v_old = v_new, p_old = p_new;
    // Eigen::VectorXd u_ana = u_new, v_ana = v_new, p_ana = p_new;
    Eigen::VectorXd grad_p_x = u_new, grad_p_y = v_new;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    double x, y;
    double Re = 40.0, Pr = 0.7, ly = 1.0, lx = 10.0, u_bulk = 1.0, T_inlet = 0.0, T_wall = 1.0;
    parameters.rho = 10.0, parameters.mu = parameters.rho * u_bulk * ly / Re;
    double therm_diff = parameters.mu / (parameters.rho * Pr);
    int tecplot_it_frequency = 100; //tecplot file written after every tecplot_it_frequency timesteps
    for (int iv = 0; iv < points.nv; iv++)
    { // Set boundary conditions here
        x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1];
        u_dirichlet_flag.push_back(true); //initialize to dirichlet
        v_dirichlet_flag.push_back(true); //initialize to dirichlet
        p_dirichlet_flag.push_back(true); //initialize to dirichlet
        // p_ana[iv] = (lx - x) * 12.0 * parameters.mu * u_bulk / (ly * ly);
        // u_ana[iv] = 6.0 * u_bulk * ((y / ly) - (y * y / (ly * ly)));
        grad_p_x[iv] = 12.0 * parameters.mu * u_bulk / (ly * ly);
        if (points.boundary_flag[iv])
            if ((fabs(y) < 1E-5) || (fabs(y - ly) < 1E-5)) //wall at top and bottom
                u_new[iv] = 0.0, v_new[iv] = 0.0, p_dirichlet_flag[iv] = false;
    }
    u_old = u_new, v_old = v_new, p_old = p_new;

    // points.calc_elem_bc_tag(parameters);
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, max_abs(u_new), max_abs(v_new), 0.0, parameters.mu / parameters.rho);

    // FRACTIONAL_STEP_1 fractional_step_1(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, temporal_order);
    SEMI_IMPLICIT_SPLIT_SOLVER semi_implicit_split_solver(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, n_outer_iter, iterative_tolerance, precond_freq_it);
    double total_steady_l1_err;
    vector<double> total_steady_l1_err_log;
    vector<string> variable_names{"u_new", "v_new", "p_new"};
    vector<Eigen::VectorXd *> variable_pointers{&u_new, &v_new, &p_new};
    if (tecplot_it_frequency > 0)
    {
        write_tecplot_temporal_variables_header(points, parameters, variable_names);
        write_tecplot_temporal_variables(points, parameters, variable_names, variable_pointers, 0);
    }
    vector<int> n_outer_iter_log;
    vector<double> iterative_l1_err_log, iterative_max_err_log;
    clock_t clock_t1 = clock(), clock_t2 = clock();
    cout << "\nTime marching started\n\n";
    for (int it = 0; it < parameters.nt; it++)
    {
        // fractional_step_1.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, grad_p_x, grad_p_y, it);
        semi_implicit_split_solver.iterative_tolerance = 0.5 * total_steady_l1_err * parameters.dt;
        if (semi_implicit_split_solver.iterative_tolerance > iterative_tolerance)
            semi_implicit_split_solver.iterative_tolerance = iterative_tolerance;
        semi_implicit_split_solver.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, grad_p_x, grad_p_y, it, n_outer_iter_log, iterative_l1_err_log, iterative_max_err_log);
        total_steady_l1_err = (u_new - u_old).lpNorm<1>() / u_bulk;
        total_steady_l1_err += ((v_new - v_old).lpNorm<1>() / u_bulk);
        total_steady_l1_err = total_steady_l1_err / ((parameters.dimension + 1.0) * parameters.dt * u_new.size());
        total_steady_l1_err_log.push_back(total_steady_l1_err);
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || total_steady_l1_err < parameters.steady_tolerance) //|| true
        {
            if (p_new.rows() == points.nv + 1)
                printf("    pressure regularization alpha: %g\n", p_new[points.nv]);
            if (iterative_max_err_log.size() > 0)
                printf("    Outer iterations: l1_error: %g, max_error: %g, iter_num: %i, tolerance: %g\n", iterative_l1_err_log[it], iterative_max_err_log[it], n_outer_iter_log[it], semi_implicit_split_solver.iterative_tolerance);
            printf("    total steady state l1_error: %g, steady_tolerance: %g\n", total_steady_l1_err, parameters.steady_tolerance);
            printf("    Completed it: %i of nt: %i, dt: %g, in CPU time: %g seconds\n\n", it, parameters.nt, parameters.dt, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
        if (tecplot_it_frequency > 0 && it > 0)
            if (it % tecplot_it_frequency == 0 || total_steady_l1_err < parameters.steady_tolerance || it == parameters.nt - 1)
                write_tecplot_temporal_variables(points, parameters, variable_names, variable_pointers, it + 1);
        parameters.nt_actual = it + 1;
        u_old = u_new, v_old = v_new, p_old = p_new;
        if (total_steady_l1_err < parameters.steady_tolerance && it > 1)
            break;
    }
    parameters.solve_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    printf("Time marching ended; factoring_timer: %g, solve_timer:%g seconds\n\n", parameters.factoring_timer, parameters.solve_timer);

    calc_navier_stokes_residuals_2D(points, parameters, u_new, v_new, p_new, grad_p_x, grad_p_y);
    parameters.total_timer = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    write_simulation_details(points, cloud, parameters), write_iteration_details(parameters);
    write_navier_stokes_residuals_2D(points, parameters, u_new, v_new, p_new, "_residuals_new.csv");

    FILE *file;
    string output_file = parameters.output_file_prefix + "_steady_error.csv";
    file = fopen(output_file.c_str(), "w");
    fprintf(file, "time(s),l1_error\n");
    for (int it = 0; it < total_steady_l1_err_log.size(); it++)
        fprintf(file, "%.16g,%.16g\n", it * parameters.dt, total_steady_l1_err_log[it]);
    fclose(file);
}