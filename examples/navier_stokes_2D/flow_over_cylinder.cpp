//Author: Dr. Shantanu Shahane
//compile: time make flow_over_cylinder
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
#include "../../header_files/coefficient_computations.hpp"
using namespace std;

void write_interp_csv(POINTS &points, PARAMETERS &parameters, vector<double> &xyz_interp, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p, Eigen::VectorXd &vorticity, const char *file_name)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_interp = calc_interp_matrix(xyz_interp, points, parameters);
    Eigen::VectorXd u_interp = mat_interp * u;
    Eigen::VectorXd v_interp = mat_interp * v;
    Eigen::VectorXd p_interp = mat_interp * p;
    Eigen::VectorXd vort_interp = mat_interp * vorticity;
    int dim = parameters.dimension;
    FILE *file;
    file = fopen(file_name, "w");
    fprintf(file, "x,y,u,v,p,vorticity\n");
    for (int ir = 0; ir < u_interp.size(); ir++)
        fprintf(file, "%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n", xyz_interp[dim * ir], xyz_interp[dim * ir + 1], u_interp(ir), v_interp(ir), p_interp(ir), vort_interp(ir));
    fclose(file);
    mat_interp.resize(0, 0), u_interp.resize(0), v_interp.resize(0), p_interp.resize(0);
}

void calc_cd_cl(vector<double> &cd, vector<double> &cl, PARAMETERS &parameters, Eigen::VectorXd &p_interp, Eigen::VectorXd &sigma_xx_interp, Eigen::VectorXd &sigma_xy_interp, Eigen::VectorXd &sigma_yy_interp, vector<double> &theta_list, double diameter, double u_inf)
{
    if (theta_list.size() % 2 != 0)
    {
        cout << "\n\nError from calc_cd_cl theta_list should be of even size, actual size: " << theta_list.size() << "\n\n";
        throw bad_exception();
    }
    double lift = 0.0, drag = 0.0, pressure = 0.0, del_theta = theta_list[1] - theta_list[0], temp_x, temp_y, cos_t, sin_t;
    for (int i1 = 0; i1 < theta_list.size(); i1++)
    {
        cos_t = cos(theta_list[i1]), sin_t = sin(theta_list[i1]);
        temp_x = (0.5 * diameter * ((-p_interp[i1] + sigma_xx_interp[i1]) * cos_t + sigma_xy_interp[i1] * sin_t));
        temp_y = (0.5 * diameter * ((-p_interp[i1] + sigma_yy_interp[i1]) * sin_t + sigma_xy_interp[i1] * cos_t));
        if (i1 % 2 == 0)
        { //simpson's rule: multiplied by 2
            lift = lift + (2.0 * temp_y);
            drag = drag + (2.0 * temp_x);
        }
        else
        { //simpson's rule: multiplied by 4
            lift = lift + (4.0 * temp_y);
            drag = drag + (4.0 * temp_x);
        }
    }
    lift = lift * del_theta / 3.0, drag = drag * del_theta / 3.0; //simpson's rule
    lift = lift / (parameters.rho * u_inf * u_inf * 0.5 * diameter);
    drag = drag / (parameters.rho * u_inf * u_inf * 0.5 * diameter);
    cl.push_back(lift), cd.push_back(drag);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    PARAMETERS parameters("parameters_file.csv", "../../gmsh_files/hole_geometries/circle_in_rectangle/up_20_down_30_ly_20_nc_40_cof_5.msh");
    int dim = parameters.dimension, temporal_order = 2;
    double iterative_tolerance = 1E-5; //parameters.steady_tolerance;
    int precond_freq_it = 10000, n_outer_iter = 10;

    POINTS points(parameters);
    CLOUD cloud(points, parameters);
    Eigen::VectorXd p_new = Eigen::VectorXd::Zero(points.nv);
    Eigen::VectorXd u_new = Eigen::VectorXd::Zero(points.nv), v_new = u_new;
    Eigen::VectorXd u_old = u_new, v_old = v_new, vorticity = u_new, p_old = p_new;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    double x, y, r;
    FILE *file;
    string file_name;

    /////////// Modify this section
    double Re = 20.0, diameter = 1.0, u_bulk = 1.0;
    double lx_up = 20 * diameter, lx_down = 30 * diameter, ly = 20 * diameter;
    parameters.rho = 1.0, parameters.mu = parameters.rho * u_bulk * diameter / Re;
    int tecplot_it_frequency = 1000; //tecplot file written after every tecplot_it_frequency timesteps
    for (int iv = 0; iv < points.nv; iv++)
    { // Set boundary conditions here
        x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1], r = sqrt(x * x + y * y);
        u_dirichlet_flag.push_back(true); //initialize to dirichlet
        v_dirichlet_flag.push_back(true); //initialize to dirichlet
        p_dirichlet_flag.push_back(true); //initialize to dirichlet
        u_new[iv] = u_bulk;               //initialize entire domain to bulk velocity
        if (points.boundary_flag[iv])
        {
            if (fabs(r - (0.5 * diameter)) < 1E-5) //dirichlet BC along cylinder
                u_new[iv] = 0.0, v_new[iv] = 0.0, u_dirichlet_flag[iv] = true, v_dirichlet_flag[iv] = true, p_dirichlet_flag[iv] = false;
            if (fabs(x + lx_up) < 1E-5) //dirichlet BC at inlet
                u_new[iv] = u_bulk, v_new[iv] = 0.0, u_dirichlet_flag[iv] = true, v_dirichlet_flag[iv] = true, p_dirichlet_flag[iv] = false;
            if (fabs(x - lx_down) < 1E-5) //outlet BC
                p_dirichlet_flag[iv] = true;
            if (fabs(fabs(y) - ly) < 1E-5)
            {                                                 //symmetry: zero shear stress
                u_dirichlet_flag[iv] = false;                 //neumann at top and bottom
                v_new[iv] = 0.0, v_dirichlet_flag[iv] = true; //dirichlet at top and bottom
                p_dirichlet_flag[iv] = false;
            }
        }
    }
    u_old = u_new, v_old = v_new;
    ///////////

    int n_theta = 360;
    vector<double> theta_list, xy_r_1;
    for (int i1 = 0; i1 < n_theta; i1++)
    {
        theta_list.push_back(i1 * 2.0 * M_PI / n_theta);
        xy_r_1.push_back(0.5 * diameter * cos(theta_list[i1])), xy_r_1.push_back(0.5 * diameter * sin(theta_list[i1]));
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor> xy_r_1_mat = calc_interp_matrix(xy_r_1, points, parameters);
    vector<double> drag_coeff, lift_coeff;
    Eigen::VectorXd p_interp, sigma_xx_interp, sigma_xy_interp, sigma_yy_interp;

    Eigen::MatrixXd temporal_p_r_1 = -Eigen::MatrixXd::Ones(100, n_theta), temporal_vorticity_r_1 = -Eigen::MatrixXd::Ones(100, n_theta);
    vector<double> temporal_xy;
    for (int ir = 1; ir <= 6; ir++)
        for (int i1 = 0; i1 < 8; i1++)
            temporal_xy.push_back(ir * 0.5 * diameter * cos(i1 * M_PI / 4.0)), temporal_xy.push_back(ir * 0.5 * diameter * sin(i1 * M_PI / 4.0));
    file_name = parameters.output_file_prefix + "_temporal_xy.csv";
    write_csv_xyz(temporal_xy, parameters, file_name.c_str());
    Eigen::VectorXd temporal_interp_data;
    Eigen::SparseMatrix<double, Eigen::RowMajor> temporal_xy_mat = calc_interp_matrix(temporal_xy, points, parameters);
    write_csv_temporal_data_init(temporal_xy_mat.rows(), (parameters.output_file_prefix + "_temporal_u.csv").c_str());
    write_csv_temporal_data_init(temporal_xy_mat.rows(), (parameters.output_file_prefix + "_temporal_v.csv").c_str());
    write_csv_temporal_data_init(temporal_xy_mat.rows(), (parameters.output_file_prefix + "_temporal_p.csv").c_str());
    write_csv_temporal_data_init(temporal_xy_mat.rows(), (parameters.output_file_prefix + "_temporal_vorticity.csv").c_str());
    temporal_interp_data = temporal_xy_mat * u_new;
    write_csv_temporal_data(temporal_interp_data, 0.0 * parameters.dt, (parameters.output_file_prefix + "_temporal_u.csv").c_str());
    temporal_interp_data = temporal_xy_mat * v_new;
    write_csv_temporal_data(temporal_interp_data, 0.0 * parameters.dt, (parameters.output_file_prefix + "_temporal_v.csv").c_str());
    temporal_interp_data = temporal_xy_mat * p_new;
    write_csv_temporal_data(temporal_interp_data, 0.0 * parameters.dt, (parameters.output_file_prefix + "_temporal_p.csv").c_str());
    vorticity = (points.grad_x_matrix_EIGEN * v_new) - (points.grad_y_matrix_EIGEN * u_new);
    temporal_interp_data = temporal_xy_mat * vorticity;
    write_csv_temporal_data(temporal_interp_data, 0.0 * parameters.dt, (parameters.output_file_prefix + "_temporal_vorticity.csv").c_str());

    points.calc_elem_bc_tag(parameters);
    if (parameters.Courant > 0.0)
        parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, max_abs(u_new), max_abs(v_new), 0.0, parameters.mu / parameters.rho);
    else
        parameters.dt = 0.05;

    // FRACTIONAL_STEP_1 fractional_step_1(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, temporal_order);
    SEMI_IMPLICIT_SPLIT_SOLVER semi_implicit_split_solver(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, n_outer_iter, iterative_tolerance, precond_freq_it);
    vector<int> n_outer_iter_log;
    vector<double> iterative_l1_err_log, iterative_max_err_log;
    double total_steady_l1_err;
    vector<double> total_steady_l1_err_log;
    vector<string> variable_names{"u_new", "v_new", "p_new", "vorticity"};
    vector<Eigen::VectorXd *> variable_pointers{&u_new, &v_new, &p_new, &vorticity};
    if (tecplot_it_frequency > 0)
    {
        write_tecplot_temporal_variables_header(points, parameters, variable_names);
        write_tecplot_temporal_variables(points, parameters, variable_names, variable_pointers, 0);
    }
    clock_t clock_t1 = clock(), clock_t2 = clock();
    cout << "\nTime marching started\n\n";
    for (int it = 0; it < parameters.nt; it++)
    {
        // fractional_step_1.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, it);
        semi_implicit_split_solver.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, it, n_outer_iter_log, iterative_l1_err_log, iterative_max_err_log);
        vorticity = (points.grad_x_matrix_EIGEN * v_new) - (points.grad_y_matrix_EIGEN * u_new);
        sigma_xx_interp = xy_r_1_mat * (2.0 * parameters.mu * (points.grad_x_matrix_EIGEN * u_new));
        sigma_yy_interp = xy_r_1_mat * (2.0 * parameters.mu * (points.grad_y_matrix_EIGEN * v_new));
        sigma_xy_interp = xy_r_1_mat * (parameters.mu * (points.grad_y_matrix_EIGEN * u_new + points.grad_x_matrix_EIGEN * v_new));
        p_interp = xy_r_1_mat * p_new;
        calc_cd_cl(drag_coeff, lift_coeff, parameters, p_interp, sigma_xx_interp, sigma_xy_interp, sigma_yy_interp, theta_list, diameter, u_bulk);
        total_steady_l1_err = (u_new - u_old).lpNorm<1>() / (u_new.lpNorm<Eigen::Infinity>());
        total_steady_l1_err += ((v_new - v_old).lpNorm<1>() / (v_new.lpNorm<Eigen::Infinity>()));
        total_steady_l1_err = total_steady_l1_err / (parameters.dimension * parameters.dt * u_new.size());
        total_steady_l1_err_log.push_back(total_steady_l1_err);
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || total_steady_l1_err < parameters.steady_tolerance) //|| true
        {
            // printf("    pressure regularization alpha: %g\n", p_new[points.nv]);
            if (iterative_max_err_log.size() > 0)
                printf("    Outer iterations: l1_error: %g, max_error: %g, iter_num: %i\n", iterative_l1_err_log[it], iterative_max_err_log[it], n_outer_iter_log[it]);
            printf("    lift coeff: %g, drag coeff: %g\n", lift_coeff[it], drag_coeff[it]);
            printf("    total steady state l1_error: %g, steady_tolerance: %g\n", total_steady_l1_err, parameters.steady_tolerance);
            printf("    Completed it: %i of nt: %i, dt: %g, in CPU time: %g seconds\n\n", it, parameters.nt, parameters.dt, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
        if (tecplot_it_frequency > 0)
            if (it % tecplot_it_frequency == 0 || total_steady_l1_err < parameters.steady_tolerance || it == parameters.nt - 1)
                write_tecplot_temporal_variables(points, parameters, variable_names, variable_pointers, it + 1);
        parameters.nt_actual = it + 1;
        u_old = u_new, v_old = v_new, p_old = p_new;
        temporal_interp_data = temporal_xy_mat * u_new;
        write_csv_temporal_data(temporal_interp_data, (it + 1.0) * parameters.dt, (parameters.output_file_prefix + "_temporal_u.csv").c_str());
        temporal_interp_data = temporal_xy_mat * v_new;
        write_csv_temporal_data(temporal_interp_data, (it + 1.0) * parameters.dt, (parameters.output_file_prefix + "_temporal_v.csv").c_str());
        temporal_interp_data = temporal_xy_mat * p_new;
        write_csv_temporal_data(temporal_interp_data, (it + 1.0) * parameters.dt, (parameters.output_file_prefix + "_temporal_p.csv").c_str());
        temporal_interp_data = temporal_xy_mat * vorticity;
        write_csv_temporal_data(temporal_interp_data, (it + 1.0) * parameters.dt, (parameters.output_file_prefix + "_temporal_vorticity.csv").c_str());
        if (total_steady_l1_err < parameters.steady_tolerance && it > 1)
            break;
        if (Re > 40.0 && (it * parameters.dt > 200.0))
            break;
    }
    parameters.solve_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    printf("Time marching ended; factoring_timer: %g, solve_timer:%g seconds\n\n", parameters.factoring_timer, parameters.solve_timer);

    calc_navier_stokes_residuals_2D(points, parameters, u_new, v_new, p_new);
    parameters.total_timer = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    write_simulation_details(points, cloud, parameters), write_iteration_details(parameters);
    write_navier_stokes_residuals_2D(points, parameters, u_new, v_new, p_new, "_residuals_new.csv");

    string output_file = parameters.output_file_prefix + "_steady_error.csv";
    file = fopen(output_file.c_str(), "w");
    fprintf(file, "time(s),l1_error\n");
    for (int it = 0; it < total_steady_l1_err_log.size(); it++)
        fprintf(file, "%.16g,%.16g\n", it * parameters.dt, total_steady_l1_err_log[it]);
    fclose(file);
    output_file = parameters.output_file_prefix + "_lift_drag.csv";
    file = fopen(output_file.c_str(), "w");
    fprintf(file, "time(s),lift,drag\n");
    for (int it = 0; it < lift_coeff.size(); it++)
        fprintf(file, "%.16g,%.16g,%.16g\n", it * parameters.dt, lift_coeff[it], drag_coeff[it]);
    fclose(file);

    if (Re <= 50)
    {
        vector<double> xy;
        double rad = 0.5 * diameter;
        for (int ir = 1; ir <= 6; ir++)
        {
            xy.clear();
            for (int i1 = 0; i1 < n_theta; i1++)
                xy.push_back(ir * rad * cos(theta_list[i1])), xy.push_back(ir * rad * sin(theta_list[i1]));
            file_name = parameters.output_file_prefix + "_ir_" + to_string(ir) + ".csv";
            write_interp_csv(points, parameters, xy, u_new, v_new, p_new, vorticity, file_name.c_str());
        }
        double delta_r = 0.02;
        for (int i1 = 0; i1 <= 7; i1++)
        {
            xy.clear();
            rad = 0.5 * diameter;
            while (true)
            {
                xy.push_back(rad * cos(i1 * M_PI / 4.0));
                xy.push_back(rad * sin(i1 * M_PI / 4.0));
                if (rad > 3 * diameter)
                    break;
                rad = rad + delta_r;
            }
            file_name = parameters.output_file_prefix + "_theta_" + to_string(i1 * 45) + ".csv";
            write_interp_csv(points, parameters, xy, u_new, v_new, p_new, vorticity, file_name.c_str());
        }
    }
}