//Author: Dr. Shantanu Shahane
//compile: time make bell_colella_JCP
//execute: time ./out
//reference: Bell JB, Colella P, Glaz HM. A second-order projection method for the incompressible Navier-Stokes equations. Journal of Computational Physics. 1989 Dec 1;85(2):257-83.
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
#include "../../header_files/coefficient_computations.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    int nx = 32;
    string meshfile = "/media/shantanu/Data/All Simulation Results/Meshless_Methods/CAD_mesh_files/Square/gmsh/Square_n_" + to_string(nx) + "_unstruc.msh";
    PARAMETERS parameters("parameters_file.csv", meshfile);

    int dim = parameters.dimension, temporal_order = 1;
    double Re = 100.0;
    parameters.rho = 1.0, parameters.mu = parameters.rho / Re;
    double x, y, nu = parameters.mu / parameters.rho, t_end = 0.5, total_steady_l1_err;
    POINTS points(parameters);
    CLOUD cloud(points, parameters);
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, 1.0, 1.0, 0.0, parameters.mu / parameters.rho);
    double dt_stable = parameters.dt;
    parameters.dt = 0.01; //1.0 / (nx * nx);
    // if (parameters.dt > dt_stable)
    // {
    //     cout << "\n\nError from main stability dt: " << dt_stable << ", set dt: " << parameters.dt << "\n\n";
    //     throw bad_exception();
    // }
    parameters.Courant = parameters.Courant * parameters.dt / dt_stable;
    cout << "Set dt: " << parameters.dt << ", Co: " << parameters.Courant << ", nt till t_end : " << t_end / parameters.dt << "\n\n ";

    Eigen::VectorXd p_new,
        p_ana;
    p_new = Eigen::VectorXd::Zero(points.nv + 1);
    Eigen::VectorXd u_new = Eigen::VectorXd::Zero(points.nv), v_new = u_new;
    Eigen::VectorXd u_old = u_new, v_old = v_new, p_old = p_new;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    for (int iv = 0; iv < points.nv; iv++)
    {
        u_dirichlet_flag.push_back(true);  //initialize to dirichlet
        v_dirichlet_flag.push_back(true);  //initialize to dirichlet
        p_dirichlet_flag.push_back(false); //initialize to neumann
        x = points.xyz[parameters.dimension * iv], y = points.xyz[parameters.dimension * iv + 1];
        u_new[iv] = -sin(M_PI * x) * sin(M_PI * x) * sin(2.0 * M_PI * y);
        v_new[iv] = sin(M_PI * y) * sin(M_PI * y) * sin(2.0 * M_PI * x);
    }
    u_old = u_new, v_old = v_new;
    // FRACTIONAL_STEP_1 fractional_step_1(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, temporal_order);
    vector<int> n_outer_iter_log;
    vector<double> iterative_l1_err_log, iterative_max_err_log;
    double iterative_tolerance = 1E-6; //parameters.steady_tolerance;
    int precond_freq_it = 10000, n_outer_iter = 10;
    SEMI_IMPLICIT_SPLIT_SOLVER semi_implicit_split_solver(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, n_outer_iter, iterative_tolerance, precond_freq_it);
    vector<string> variable_names{"u_new", "v_new", "p_new"};
    vector<Eigen::VectorXd *> variable_pointers{&u_new, &v_new, &p_new};
    write_tecplot_temporal_variables_header(points, parameters, variable_names);
    write_tecplot_temporal_variables(points, parameters, variable_names, variable_pointers, 0);

    clock_t clock_t1 = clock(), clock_t2 = clock();
    cout << "\nTime marching started\n\n";
    for (int it = 0; it < parameters.nt; it++)
    {
        // fractional_step_1.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, it);
        semi_implicit_split_solver.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, it, n_outer_iter_log, iterative_l1_err_log, iterative_max_err_log);
        total_steady_l1_err = (u_new - u_old).lpNorm<1>() / (u_new.lpNorm<Eigen::Infinity>());
        total_steady_l1_err += ((v_new - v_old).lpNorm<1>() / (v_new.lpNorm<Eigen::Infinity>()));
        total_steady_l1_err = total_steady_l1_err / (parameters.dimension * parameters.dt * u_new.size());
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || ((it + 1.0) * parameters.dt >= t_end)) //|| true
        {
            printf("    pressure regularization alpha: %g\n", p_new[points.nv]);
            if (iterative_max_err_log.size() > 0)
                printf("    Outer iterations: l1_error: %g, max_error: %g, iter_num: %i\n", iterative_l1_err_log[it], iterative_max_err_log[it], n_outer_iter_log[it]);
            printf("    total steady state l1_error: %g, steady_tolerance: %g\n", total_steady_l1_err, parameters.steady_tolerance);
            printf("    Completed it: %i, dt: %g, physical time: %g, in CPU time: %g seconds\n\n", it, parameters.dt, (it + 1.0) * parameters.dt, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
        parameters.nt_actual = it + 1;
        u_old = u_new, v_old = v_new, p_old = p_new;
        if ((it + 1.0) * parameters.dt >= t_end)
        {
            write_tecplot_temporal_variables(points, parameters, variable_names, variable_pointers, it + 1);
            break;
        }
    }
    parameters.solve_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    printf("Time marching ended; factoring_timer: %g, solve_timer:%g seconds\n\n", parameters.factoring_timer, parameters.solve_timer);
    write_simulation_details(points, cloud, parameters);

    int nxy = 101;
    double dx = 1.0 / (nxy - 1.0);
    vector<double> xy_interp;
    for (int i1 = 0; i1 < nxy; i1++) //horizontal line
        xy_interp.push_back(i1 * dx), xy_interp.push_back(0.5);
    for (int i1 = 0; i1 < nxy; i1++) //vertical line
        xy_interp.push_back(0.5), xy_interp.push_back(i1 * dx);
    Eigen::SparseMatrix<double, Eigen::RowMajor> interp_mat = calc_interp_matrix(xy_interp, points, parameters);
    Eigen::VectorXd u_interp = interp_mat * u_new;
    Eigen::VectorXd v_interp = interp_mat * v_new;
    Eigen::VectorXd p_interp = interp_mat * p_new.head(points.nv);
    FILE *file;
    file = fopen((parameters.output_file_prefix + "_uvp.csv").c_str(), "w");
    fprintf(file, "x,y,u,v,p\n");
    for (int ir = 0; ir < u_interp.size(); ir++)
        fprintf(file, "%.16g,%.16g,%.16g,%.16g,%.16g\n", xy_interp[dim * ir], xy_interp[dim * ir + 1], u_interp(ir), v_interp(ir), p_interp(ir));
    fclose(file);
}
