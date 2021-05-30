//Author: Dr. Shantanu Shahane
//compile: time make bellow_periodic
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
#include "../../header_files/coefficient_computations.hpp"
using namespace std;

void write_interp_csv(POINTS &points, PARAMETERS &parameters, vector<double> &xyz_interp, Eigen::VectorXd &field, Eigen::SparseMatrix<double, Eigen::RowMajor> &interp_mat, string filename, int it)
{
    Eigen::VectorXd interp_field = interp_mat * (field.head(points.nv));
    int dim = parameters.dimension;
    FILE *file;
    if (it == 0)
    {
        file = fopen(filename.c_str(), "w");
        for (int iv = 0; iv < int(xyz_interp.size() / dim); iv++)
            fprintf(file, ",%.16g", xyz_interp[iv * dim]);
        fprintf(file, "\n");
        for (int iv = 0; iv < int(xyz_interp.size() / dim); iv++)
            fprintf(file, ",%.16g", xyz_interp[iv * dim + 1]);
        fprintf(file, "\n");
        fclose(file);
    }
    file = fopen(filename.c_str(), "a");
    fprintf(file, "%.16g", it * parameters.dt);
    for (int iv = 0; iv < interp_field.size(); iv++)
        fprintf(file, ",%.16g", interp_field[iv]);
    fprintf(file, "\n");
    fclose(file);
}

void write_interp_csv_wrapper(POINTS &points, PARAMETERS &parameters, vector<double> &xyz_interp, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p, Eigen::VectorXd &vort, Eigen::SparseMatrix<double, Eigen::RowMajor> &interp_mat, string filename, int it)
{
    write_interp_csv(points, parameters, xyz_interp, u, interp_mat, filename + "_u.csv", it);
    write_interp_csv(points, parameters, xyz_interp, v, interp_mat, filename + "_v.csv", it);
    write_interp_csv(points, parameters, xyz_interp, p, interp_mat, filename + "_p.csv", it);
    write_interp_csv(points, parameters, xyz_interp, vort, interp_mat, filename + "_vort.csv", it);
}

void calc_u_avg(POINTS &points, PARAMETERS &parameters, vector<double> &xyz_interp, Eigen::VectorXd &u, Eigen::SparseMatrix<double, Eigen::RowMajor> &interp_mat, vector<double> &u_avg_bellow, vector<double> &Re_bellow, double l_charac)
{
    if (interp_mat.rows() % 2 == 0)
    {
        cout << "\n\nError from calc_u_avg interp_mat.rows() should be of odd size, actual size: " << interp_mat.rows() << "\n\n";
        throw bad_exception();
    }
    int n = interp_mat.rows();
    Eigen::VectorXd interp_field = interp_mat * u;
    double dy = xyz_interp[3] - xyz_interp[1], u_avg = interp_field[0] + interp_field[n - 1];
    for (int i1 = 1; i1 < n - 1; i1++)
    {
        if (i1 % 2 == 0) //simpson's rule: multiplied by 2
            u_avg = u_avg + (2.0 * interp_field[i1]);
        else //simpson's rule: multiplied by 4
            u_avg = u_avg + (4.0 * interp_field[i1]);
    }
    u_avg = u_avg * dy / 3.0;
    u_avg_bellow.push_back(u_avg);
    Re_bellow.push_back(parameters.rho * u_avg * l_charac / parameters.mu);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    PARAMETERS parameters("parameters_file.csv", "/media/shantanu/Data/All Simulation Results/Meshless_Methods/CAD_mesh_files/bellows/h_0p5_lambda_3_amp_0p3_lx_0_nb_1_n_15.msh");
    // PARAMETERS parameters("parameters_file.csv", "/media/shantanu/Data/All Simulation Results/Meshless_Methods/CAD_mesh_files/bellows/h_0p6_lambda_2p8_amp_0p35_lx_0_nb_1_n_15.msh");
    int dim = parameters.dimension, temporal_order = 2;
    double iterative_tolerance = 1E-4; //parameters.steady_tolerance;
    int precond_freq_it = 10000, n_outer_iter = 10;

    POINTS points(parameters);
    vector<string> periodic_axis{"x"};
    points.set_periodic_bc(parameters, periodic_axis);
    CLOUD cloud(points, parameters);

    Eigen::VectorXd p_new = Eigen::VectorXd::Zero(points.nv);
    Eigen::VectorXd u_new = Eigen::VectorXd::Zero(points.nv), v_new = u_new, T_new = u_new;
    Eigen::VectorXd u_old = u_new, v_old = v_new, p_old = p_new, T_old = u_new;
    Eigen::VectorXd grad_p_x = u_new, grad_p_y = v_new, vorticity = u_new;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, T_dirichlet_flag;
    double x, y, physical_time;
    double Re = 50.0, ly = 0.5, u_bulk = 1.0, dxy_interp = 0.01, lambda = 3.0, amp = 0.3, l_in_out = 0.0;
    parameters.rho = 1.0, parameters.mu = parameters.rho * u_bulk * ly / Re;
    int tecplot_it_frequency = 100; //tecplot file written after every tecplot_it_frequency timesteps
    for (int iv = 0; iv < points.nv; iv++)
    { // Set boundary conditions here
        x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1];
        u_dirichlet_flag.push_back(true); //initialize to dirichlet
        v_dirichlet_flag.push_back(true); //initialize to dirichlet
        p_dirichlet_flag.push_back(true); //initialize to dirichlet
        T_dirichlet_flag.push_back(true); //initialize to dirichlet
        grad_p_x[iv] = 12.0 * parameters.mu * u_bulk / (ly * ly);
        if (points.boundary_flag[iv])
            if (y > 0)
                u_new[iv] = 0.0, v_new[iv] = 0.0, p_dirichlet_flag[iv] = false, T_new[iv] = 1.0;
            else
                u_new[iv] = 0.0, v_new[iv] = 0.0, p_dirichlet_flag[iv] = false, T_new[iv] = -1.0;
    }
    u_old = u_new, v_old = v_new, p_old = p_new, T_old = T_new;
    vorticity = (points.grad_x_matrix_EIGEN * v_new) - (points.grad_y_matrix_EIGEN * u_new);
    vector<double> xy_horz, xy_vert_0p25, xy_vert_0p5, xy_vert_0p75, xy_vert_0, u_avg_bellow, Re_bellow;
    x = 0.0, y = 0.0;
    while (x <= lambda + (2.0 * l_in_out))
        xy_horz.push_back(x), xy_horz.push_back(y), x = x + dxy_interp;
    x = l_in_out, y = -(-amp * cos(2.0 * M_PI * (x - l_in_out) / lambda) + (0.5 * ly) + amp);
    while (fabs(y) <= (-amp * cos(2.0 * M_PI * (x - l_in_out) / lambda) + (0.5 * ly) + amp))
        xy_vert_0.push_back(x), xy_vert_0.push_back(y), y = y + (ly / 101);
    x = l_in_out + (0.25 * lambda), y = -(-amp * cos(2.0 * M_PI * (x - l_in_out) / lambda) + (0.5 * ly) + amp);
    while (fabs(y) <= (-amp * cos(2.0 * M_PI * (x - l_in_out) / lambda) + (0.5 * ly) + amp))
        xy_vert_0p25.push_back(x), xy_vert_0p25.push_back(y), y = y + dxy_interp;
    x = l_in_out + (0.5 * lambda), y = -(-amp * cos(2.0 * M_PI * (x - l_in_out) / lambda) + (0.5 * ly) + amp);
    while (fabs(y) <= (-amp * cos(2.0 * M_PI * (x - l_in_out) / lambda) + (0.5 * ly) + amp))
        xy_vert_0p5.push_back(x), xy_vert_0p5.push_back(y), y = y + dxy_interp;
    x = l_in_out + (0.75 * lambda), y = -(-amp * cos(2.0 * M_PI * (x - l_in_out) / lambda) + (0.5 * ly) + amp);
    while (fabs(y) <= (-amp * cos(2.0 * M_PI * (x - l_in_out) / lambda) + (0.5 * ly) + amp))
        xy_vert_0p75.push_back(x), xy_vert_0p75.push_back(y), y = y + dxy_interp;
    Eigen::SparseMatrix<double, Eigen::RowMajor> interp_mat_xy_horz = calc_interp_matrix(xy_horz, points, parameters);
    Eigen::SparseMatrix<double, Eigen::RowMajor> interp_mat_xy_vert_0 = calc_interp_matrix(xy_vert_0, points, parameters);
    Eigen::SparseMatrix<double, Eigen::RowMajor> interp_mat_xy_vert_0p25 = calc_interp_matrix(xy_vert_0p25, points, parameters);
    Eigen::SparseMatrix<double, Eigen::RowMajor> interp_mat_xy_vert_0p5 = calc_interp_matrix(xy_vert_0p5, points, parameters);
    Eigen::SparseMatrix<double, Eigen::RowMajor> interp_mat_xy_vert_0p75 = calc_interp_matrix(xy_vert_0p75, points, parameters);
    write_interp_csv_wrapper(points, parameters, xy_horz, u_new, v_new, p_new, vorticity, interp_mat_xy_horz, parameters.output_file_prefix + "_xy_horz", 0);
    write_interp_csv_wrapper(points, parameters, xy_vert_0p25, u_new, v_new, p_new, vorticity, interp_mat_xy_vert_0p25, parameters.output_file_prefix + "xy_vert_0p25", 0);
    write_interp_csv_wrapper(points, parameters, xy_vert_0p5, u_new, v_new, p_new, vorticity, interp_mat_xy_vert_0p5, parameters.output_file_prefix + "xy_vert_0p5", 0);
    write_interp_csv_wrapper(points, parameters, xy_vert_0p75, u_new, v_new, p_new, vorticity, interp_mat_xy_vert_0p75, parameters.output_file_prefix + "xy_vert_0p75", 0);
    calc_u_avg(points, parameters, xy_vert_0, u_new, interp_mat_xy_vert_0, u_avg_bellow, Re_bellow, ly);

    // points.calc_elem_bc_tag(parameters);
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, 1.0, 1.0, 0.0, parameters.mu / parameters.rho);

    // FRACTIONAL_STEP_1 fractional_step_1(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, temporal_order);
    SEMI_IMPLICIT_SPLIT_SOLVER semi_implicit_split_solver(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, n_outer_iter, iterative_tolerance, precond_freq_it);
    double unsteady_coeff = 1.0 / parameters.dt, conv_coeff = 1.0, diff_coeff = -parameters.mu / (parameters.rho * 0.7);
    IMPLICIT_SCALAR_TRANSPORT_SOLVER implicit_scalar_transport_solver(points, cloud, parameters, T_dirichlet_flag, precond_freq_it, unsteady_coeff, conv_coeff, diff_coeff, false);
    double total_steady_l1_err;
    vector<double> total_steady_l1_err_log;
    vector<string> variable_names{"u_new", "v_new", "p_new", "vorticity", "T_new"};
    vector<Eigen::VectorXd *> variable_pointers{&u_new, &v_new, &p_new, &vorticity, &T_new};
    write_tecplot_temporal_variables_header(points, parameters, variable_names);
    write_tecplot_temporal_variables(points, parameters, variable_names, variable_pointers, 0);
    vector<int> n_outer_iter_log;
    vector<double> iterative_l1_err_log, iterative_max_err_log;
    clock_t clock_t1 = clock(), clock_t2 = clock();
    cout << "\nTime marching started\n\n";
    for (int it = 0; it < parameters.nt; it++)
    {
        physical_time = (it + 1.0) * parameters.dt;
        // fractional_step_1.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, grad_p_x, grad_p_y, it);
        // semi_implicit_split_solver.iterative_tolerance = 0.5 * total_steady_l1_err * parameters.dt;
        semi_implicit_split_solver.iterative_tolerance = 0.5 * total_steady_l1_err * parameters.dt;
        if (semi_implicit_split_solver.iterative_tolerance > iterative_tolerance)
            semi_implicit_split_solver.iterative_tolerance = iterative_tolerance;
        semi_implicit_split_solver.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, grad_p_x, grad_p_y, it, n_outer_iter_log, iterative_l1_err_log, iterative_max_err_log);
        implicit_scalar_transport_solver.single_timestep_2d(points, cloud, parameters, T_new, T_old, u_new, v_new, it);
        vorticity = (points.grad_x_matrix_EIGEN * v_new) - (points.grad_y_matrix_EIGEN * u_new);
        total_steady_l1_err = (u_new - u_old).lpNorm<1>();
        total_steady_l1_err += (v_new - v_old).lpNorm<1>();
        total_steady_l1_err += (T_new - T_old).lpNorm<1>();
        total_steady_l1_err = total_steady_l1_err / ((parameters.dimension + 1.0) * parameters.dt * u_new.size());
        total_steady_l1_err_log.push_back(total_steady_l1_err);
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;

        if ((it + 1) % 1 == 0)
        {
            write_interp_csv_wrapper(points, parameters, xy_horz, u_new, v_new, p_new, vorticity, interp_mat_xy_horz, parameters.output_file_prefix + "_xy_horz", it + 1);
            write_interp_csv_wrapper(points, parameters, xy_vert_0p25, u_new, v_new, p_new, vorticity, interp_mat_xy_vert_0p25, parameters.output_file_prefix + "xy_vert_0p25", it + 1);
            write_interp_csv_wrapper(points, parameters, xy_vert_0p5, u_new, v_new, p_new, vorticity, interp_mat_xy_vert_0p5, parameters.output_file_prefix + "xy_vert_0p5", it + 1);
            write_interp_csv_wrapper(points, parameters, xy_vert_0p75, u_new, v_new, p_new, vorticity, interp_mat_xy_vert_0p75, parameters.output_file_prefix + "xy_vert_0p75", it + 1);
        }
        calc_u_avg(points, parameters, xy_vert_0, u_new, interp_mat_xy_vert_0, u_avg_bellow, Re_bellow, ly);

        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || total_steady_l1_err < parameters.steady_tolerance) //|| true
        {
            if (p_new.rows() == points.nv + 1)
                printf("    pressure regularization alpha: %g\n", p_new[points.nv]);
            if (iterative_max_err_log.size() > 0)
                printf("    Outer iterations: l1_error: %g, max_error: %g, iter_num: %i, tolerance: %g\n", iterative_l1_err_log[it], iterative_max_err_log[it], n_outer_iter_log[it], semi_implicit_split_solver.iterative_tolerance);
            printf("    total steady state l1_error: %g, steady_tolerance: %g\n", total_steady_l1_err, parameters.steady_tolerance);
            printf("    u_max: %g, u_avg: %g, Re: %g\n", u_new.lpNorm<Eigen::Infinity>(), u_avg_bellow[it + 1], Re_bellow[it + 1]);
            printf("    Completed it: %i, nt: %i, dt: %g, physical time: %g, in CPU time: %g seconds\n\n", it, parameters.nt, parameters.dt, physical_time, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
        if (tecplot_it_frequency > 0 && it > 0)
            if (total_steady_l1_err < parameters.steady_tolerance || it == parameters.nt - 1 || (fabs((int)(0.1 * physical_time) - (0.1 * physical_time)) <= (0.1 * parameters.dt)))
                write_tecplot_temporal_variables(points, parameters, variable_names, variable_pointers, it + 1);
        parameters.nt_actual = it + 1;
        u_old = u_new, v_old = v_new, p_old = p_new, T_old = T_new;
        if (total_steady_l1_err < parameters.steady_tolerance || physical_time > 300.0)
            break;
    }
    parameters.solve_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    printf("Time marching ended; factoring_timer: %g, solve_timer:%g seconds\n\n", parameters.factoring_timer, parameters.solve_timer);
    parameters.total_timer = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    write_simulation_details(points, cloud, parameters), write_iteration_details(parameters);
    FILE *file;
    string output_file = parameters.output_file_prefix + "_steady_error.csv";
    file = fopen(output_file.c_str(), "w");
    fprintf(file, "time(s),l1_error\n");
    for (int it = 0; it < total_steady_l1_err_log.size(); it++)
        fprintf(file, "%.16g,%.16g\n", it * parameters.dt, total_steady_l1_err_log[it]);
    fclose(file);

    file = fopen((parameters.output_file_prefix + "_u_avg_Re.csv").c_str(), "w");
    fprintf(file, "time(s),u_avg,Re\n");
    for (int it = 0; it < u_avg_bellow.size(); it++)
        fprintf(file, "%.16g,%.16g,%.16g\n", it * parameters.dt, u_avg_bellow[it], Re_bellow[it]);
    fclose(file);
}
