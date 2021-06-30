//Author: Dr. Shantanu Shahane
//compile: time make kovasznay_flow
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    PARAMETERS parameters("parameters_file.csv", "../../gmsh_files/Square/Square_n_20_unstruc.msh");
    int temporal_order = 1;
    double iterative_tolerance = 1E-5;                                  //parameters.steady_tolerance;
    int precond_freq_it = 10000, n_outer_iter = 5;

    double Re = 40.0, lambda = (Re / 2) - sqrt(4 * M_PI * M_PI + (Re * Re / 4));
    parameters.rho = 1.0, parameters.mu = parameters.rho / Re;
    //for kovasznay flow, if parameters.rho!=1, the residual in X-mom is non-zero due to the dp/dx term which does not cancel. kovasznay solution defined for non-dimensional Navier-Stokes. Also, the numerical pressure does not match with analytical pressure if parameters.rho!=1. U and V still match independent of rho
    POINTS points(parameters);
    for (int i = 0; i < points.xyz.size(); i++)
        points.xyz[i] = points.xyz[i] - 0.5;
    for (int i = 0; i < points.xyz_original.size(); i++)
        points.xyz_original[i] = points.xyz_original[i] - 0.5;
    CLOUD cloud(points, parameters);
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, 1.0, 1.0, 0.0, parameters.mu / parameters.rho);

    Eigen::VectorXd p_ana, p_new;
    p_new = Eigen::VectorXd::Zero(points.nv + 1), p_ana = p_new;
    Eigen::VectorXd u_new = Eigen::VectorXd::Zero(points.nv), v_new = u_new;
    Eigen::VectorXd u_ana = u_new, v_ana = v_new;
    Eigen::VectorXd u_old = u_new, v_old = v_new, p_old = p_new;
    double x, y;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    for (int iv = 0; iv < points.nv; iv++)
    {
        u_dirichlet_flag.push_back(true);  //initialize to dirichlet
        v_dirichlet_flag.push_back(true);  //initialize to dirichlet
        p_dirichlet_flag.push_back(false); //initialize to neumann
        x = points.xyz[parameters.dimension * iv], y = points.xyz[parameters.dimension * iv + 1];
        u_ana[iv] = 1 - exp(lambda * x) * cos(2 * M_PI * y);
        v_ana[iv] = exp(lambda * x) * sin(2 * M_PI * y) * lambda / (2 * M_PI);
        p_ana[iv] = -exp(2 * lambda * x) / 2;
        if (points.boundary_flag[iv])
            u_new[iv] = u_ana[iv], v_new[iv] = v_ana[iv]; //dirichlet BC
    }
    u_old = u_new, v_old = v_new;
    // FRACTIONAL_STEP_1 fractional_step_1(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, temporal_order);
    SEMI_IMPLICIT_SPLIT_SOLVER semi_implicit_split_solver(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, n_outer_iter, iterative_tolerance, precond_freq_it);
    vector<int> n_outer_iter_log;
    vector<double> iterative_l1_err_log, iterative_max_err_log;
    double total_steady_l1_err = 1000.0;
    vector<double> total_steady_l1_err_log;
    clock_t clock_t1 = clock(), clock_t2 = clock();
    cout << "\nTime marching started\n\n";
    for (int it = 0; it < parameters.nt; it++)
    {
        // fractional_step_1.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, it);
        semi_implicit_split_solver.iterative_tolerance = 0.5 * total_steady_l1_err * parameters.dt;
        if (semi_implicit_split_solver.iterative_tolerance > iterative_tolerance)
            semi_implicit_split_solver.iterative_tolerance = iterative_tolerance;
        semi_implicit_split_solver.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, it, n_outer_iter_log, iterative_l1_err_log, iterative_max_err_log);
        total_steady_l1_err = (u_new - u_old).lpNorm<1>() / (u_new.lpNorm<Eigen::Infinity>());
        total_steady_l1_err += ((v_new - v_old).lpNorm<1>() / (v_new.lpNorm<Eigen::Infinity>()));
        total_steady_l1_err = total_steady_l1_err / (parameters.dimension * parameters.dt * u_new.size());
        total_steady_l1_err_log.push_back(total_steady_l1_err);
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || total_steady_l1_err < parameters.steady_tolerance) //|| true
        {
            printf("    pressure regularization alpha: %g\n", p_new[points.nv]);
            if (iterative_max_err_log.size() > 0)
                printf("    Outer iterations: l1_error: %g, max_error: %g, iter_num: %i, tolerance: %g\n", iterative_l1_err_log[it], iterative_max_err_log[it], n_outer_iter_log[it], semi_implicit_split_solver.iterative_tolerance);
            printf("    total steady state error: %g, steady_tolerance: %g\n", total_steady_l1_err, parameters.steady_tolerance);
            printf("    Completed it: %i of nt: %i, dt: %g, in CPU time: %g seconds\n\n", it, parameters.nt, parameters.dt, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
        parameters.nt_actual = it + 1;
        u_old = u_new, v_old = v_new, p_old = p_new;
        if (total_steady_l1_err < parameters.steady_tolerance && it > 1)
            break;
    }
    parameters.solve_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    printf("Time marching ended; factoring_timer: %g, solve_timer:%g seconds\n\n", parameters.factoring_timer, parameters.solve_timer);

    calc_navier_stokes_errors_2D(points, parameters, u_ana, v_ana, p_ana, u_new, v_new, p_new);
    calc_navier_stokes_residuals_2D(points, parameters, u_ana, v_ana, p_ana);
    printf("main: Re: %g, lambda: %g\n\n", Re, lambda);
    parameters.total_timer = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    write_simulation_details(points, cloud, parameters), write_iteration_details(parameters);
    write_navier_stokes_errors_2D(points, parameters, u_ana, v_ana, p_ana, u_new, v_new, p_new);
    write_navier_stokes_residuals_2D(points, parameters, u_ana, v_ana, p_ana, "_residuals_ana.csv");
    write_navier_stokes_residuals_2D(points, parameters, u_new, v_new, p_new, "_residuals_new.csv");
    Eigen::VectorXd u_error = (u_ana - u_new).cwiseAbs();
    Eigen::VectorXd v_error = (v_ana - v_new).cwiseAbs();
    Eigen::VectorXd p_error = (p_ana - p_new).cwiseAbs();
    vector<string> variable_names{"u_new", "v_new", "p_new", "u_ana", "v_ana", "p_ana", "u_error", "v_error", "p_error"};
    vector<Eigen::VectorXd *> variable_pointers{&u_new, &v_new, &p_new, &u_ana, &v_ana, &p_ana, &u_error, &v_error, &p_error};
    write_tecplot_steady_variables(points, parameters, variable_names, variable_pointers);

    FILE *file;
    string output_file = parameters.output_file_prefix + "_steady_error.csv";
    file = fopen(output_file.c_str(), "w");
    fprintf(file, "time(s),l1_error\n");
    for (int it = 0; it < total_steady_l1_err_log.size(); it++)
        fprintf(file, "%.16g,%.16g\n", it * parameters.dt, total_steady_l1_err_log[it]);
    fclose(file);
}
