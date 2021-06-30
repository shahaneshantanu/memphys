//Author: Dr. Shantanu Shahane
//compile: time make taylor_green_steady
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
#include "../../header_files/coefficient_computations.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    PARAMETERS parameters("parameters_file.csv", "../../gmsh_files/Square/Square_n_40_unstruc.msh");
    int dim = parameters.dimension, temporal_order = 2;
    double iterative_tolerance = 1E-5; //parameters.steady_tolerance;
    int precond_freq_it = 10000, n_outer_iter = 5;

    double Re = 100.0;
    parameters.rho = 10.0, parameters.mu = parameters.rho / Re;
    POINTS points(parameters);
    CLOUD cloud(points, parameters);

    Eigen::VectorXd p_ana, p_new, p_ana_head, p_new_head;
    p_new = Eigen::VectorXd::Zero(points.nv + 1), p_ana = p_new;
    Eigen::VectorXd u_new = Eigen::VectorXd::Zero(points.nv), v_new = u_new;
    Eigen::VectorXd u_ana = u_new, v_ana = v_new;
    Eigen::VectorXd u_old = u_new, v_old = v_new, p_old = p_new;
    Eigen::VectorXd x_mom_source = u_new, y_mom_source = v_new;
    double x, y, total_steady_err;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    for (int iv = 0; iv < points.nv; iv++)
        u_dirichlet_flag.push_back(true), v_dirichlet_flag.push_back(true), p_dirichlet_flag.push_back(false);

    for (int iv = 0; iv < points.nv; iv++)
    {
        x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1];
        u_ana[iv] = cos(x) * sin(y);
        v_ana[iv] = -sin(x) * cos(y);
        p_ana[iv] = -0.25 * parameters.rho * (cos(2 * x) + cos(2 * y));
        if (points.boundary_flag[iv])
            u_new[iv] = u_ana[iv], v_new[iv] = v_ana[iv]; //dirichlet BC
        x_mom_source[iv] = 2 * parameters.mu * cos(x) * sin(y);
        y_mom_source[iv] = -2 * parameters.mu * cos(y) * sin(x);
    }
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, max_abs(u_ana), max_abs(v_ana), 0.0, parameters.mu / parameters.rho);
    u_old = u_new, v_old = v_new;
    vector<int> n_outer_iter_log;
    vector<double> iterative_l1_err_log, iterative_max_err_log, total_steady_err_log;
    // FRACTIONAL_STEP_1 fractional_step_1(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, temporal_order);
    SEMI_IMPLICIT_SPLIT_SOLVER semi_implicit_split_solver(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, n_outer_iter, iterative_tolerance, precond_freq_it);
    clock_t clock_t1 = clock(), clock_t2 = clock();
    cout << "\nTime marching started\n\n";
    for (int it = 0; it < parameters.nt; it++)
    {
        // fractional_step_1.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, x_mom_source, y_mom_source, it);
        semi_implicit_split_solver.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, x_mom_source, y_mom_source, it, n_outer_iter_log, iterative_l1_err_log, iterative_max_err_log);
        total_steady_err = (u_new - u_old).lpNorm<1>() / (u_new.lpNorm<Eigen::Infinity>());
        total_steady_err += (v_new - v_old).lpNorm<1>() / (v_new.lpNorm<Eigen::Infinity>());
        total_steady_err = total_steady_err / (parameters.dimension * parameters.dt * u_new.size());
        total_steady_err_log.push_back(total_steady_err);
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || total_steady_err < parameters.steady_tolerance) //|| true
        {
            printf("    pressure regularization alpha: %g\n", p_new[points.nv]);
            if (iterative_max_err_log.size() > 0)
                printf("    Outer iterations: l1_error: %g, max_error: %g, iter_num: %i\n", iterative_l1_err_log[it], iterative_max_err_log[it], n_outer_iter_log[it]);
            printf("    total steady state error: %g, steady_tolerance: %g\n", total_steady_err, parameters.steady_tolerance);
            printf("    Completed it: %i of nt: %i, dt: %g, in CPU time: %g seconds\n\n", it, parameters.nt, parameters.dt, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
        parameters.nt_actual = it + 1;
        u_old = u_new, v_old = v_new, p_old = p_new;
        if (total_steady_err < parameters.steady_tolerance && it > 1)
            break;
    }

    parameters.solve_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    printf("Time marching ended; factoring_timer: %g, solve_timer:%g seconds\n\n", parameters.factoring_timer, parameters.solve_timer);

    calc_navier_stokes_errors_2D(points, parameters, u_ana, v_ana, p_ana, u_new, v_new, p_new);
    parameters.total_timer = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    write_simulation_details(points, cloud, parameters), write_iteration_details(parameters);
    write_navier_stokes_errors_2D(points, parameters, u_ana, v_ana, p_ana, u_new, v_new, p_new);

    FILE *file;
    string output_file = parameters.output_file_prefix + "_steady_error.csv";
    file = fopen(output_file.c_str(), "w");
    fprintf(file, "time(s),l1_error\n");
    for (int it = 0; it < total_steady_err_log.size(); it++)
        fprintf(file, "%.16g,%.16g\n", it * parameters.dt, total_steady_err_log[it]);
    fclose(file);
}