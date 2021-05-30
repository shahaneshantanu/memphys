//Author: Dr. Shantanu Shahane
//compile: time make couette_2_cylinders_ellipse
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
#include "../../header_files/coefficient_computations.hpp"
using namespace std;

void write_interp_csv(POINTS &points, PARAMETERS &parameters, vector<double> &xy_interp, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p, string filename)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> interp_mat = calc_interp_matrix(xy_interp, points, parameters);
    Eigen::VectorXd u_interp = interp_mat * u;
    Eigen::VectorXd v_interp = interp_mat * v;
    Eigen::VectorXd p_interp = interp_mat * p.head(points.nv);
    FILE *file;
    file = fopen(filename.c_str(), "w");
    fprintf(file, "x,y,u,v,p\n");
    for (int ir = 0; ir < u_interp.size(); ir++)
        fprintf(file, "%.16g,%.16g,%.16g,%.16g,%.16g\n", xy_interp[parameters.dimension * ir], xy_interp[parameters.dimension * ir + 1], u_interp(ir), v_interp(ir), p_interp(ir));
    fclose(file);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    PARAMETERS parameters("parameters_file.csv", "/media/shantanu/Data/All Simulation Results/Meshless_Methods/CAD_mesh_files/hole_geometries/double_circle_in_ellipse/mesh_n_100.msh");
    double iterative_tolerance = 1E-5;
    int precond_freq_it = 10000, n_outer_iter = 10;
    double Re = 40.0, omega_i = 1.0, r_i = 0.25, u_theta = omega_i * r_i;
    parameters.rho = 1.0, parameters.mu = parameters.rho * u_theta * r_i / Re;

    POINTS points(parameters);
    CLOUD cloud(points, parameters);
    clock_t t1 = clock();
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, parameters.rho, parameters.rho, 0.0, parameters.mu);

    Eigen::VectorXd p_new;
    p_new = Eigen::VectorXd::Zero(points.nv + 1);
    Eigen::VectorXd u_new = Eigen::VectorXd::Zero(points.nv), v_new = u_new;
    Eigen::VectorXd u_old = u_new, v_old = v_new, p_old = p_new;

    double x, y, r, cx_right = 0.4, cx_left = -0.4, a_ellipse = 1.0, b_ellipse = 0.75;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    for (int iv = 0; iv < points.nv; iv++)
    {
        u_dirichlet_flag.push_back(true);  //initialize to dirichlet
        v_dirichlet_flag.push_back(true);  //initialize to dirichlet
        p_dirichlet_flag.push_back(false); //initialize to neumann
        x = points.xyz[parameters.dimension * iv], y = points.xyz[parameters.dimension * iv + 1];
        if (points.boundary_flag[iv])
        { //dirichlet BC
            r = sqrt((x - cx_right) * (x - cx_right) + y * y);
            if ((r - r_i) < 1E-4) //right circle
                u_new[iv] = u_theta * (-y) / r, v_new[iv] = u_theta * (x - cx_right) / r;
            r = sqrt((x - cx_left) * (x - cx_left) + y * y);
            if ((r - r_i) < 1E-4) //left circle
                u_new[iv] = -u_theta * (-y) / r, v_new[iv] = -u_theta * (x - cx_left) / r;
        }
    }
    u_old = u_new, v_old = v_new;
    vector<string> variable_names{"u_new", "v_new", "p_new"};
    vector<Eigen::VectorXd *> variable_pointers{&u_new, &v_new, &p_new};
    write_tecplot_temporal_variables_header(points, parameters, variable_names);
    write_tecplot_temporal_variables(points, parameters, variable_names, variable_pointers, 0);

    SEMI_IMPLICIT_SPLIT_SOLVER semi_implicit_split_solver(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, n_outer_iter, iterative_tolerance, precond_freq_it);
    vector<int> n_outer_iter_log;
    vector<double> iterative_l1_err_log, iterative_max_err_log;
    double total_steady_l1_err = 1000.0;
    vector<double> total_steady_l1_err_log;
    clock_t clock_t1 = clock(), clock_t2 = clock();
    cout << "\nTime marching started\n\n";
    int it;
    for (it = 0; it < parameters.nt; it++)
    {
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
            printf("    total steady state l1_error: %g, steady_tolerance: %g\n", total_steady_l1_err, parameters.steady_tolerance);
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
    parameters.total_timer = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    write_simulation_details(points, cloud, parameters), write_iteration_details(parameters);

    double dxy = 0.005, r1, r2;
    int nxy = 1 + (2 / dxy);
    vector<double> xy_horizontal_midline;
    x = -1.0, y = 0.0;
    for (int i1 = 0; i1 < nxy; i1++)
    {
        r1 = sqrt((x - cx_right) * (x - cx_right) + y * y);
        r2 = sqrt((x - cx_left) * (x - cx_left) + y * y);
        if ((r1 >= r_i) && (r2 >= r_i))
            xy_horizontal_midline.push_back(x), xy_horizontal_midline.push_back(y);
        x = x + dxy;
    }
    write_interp_csv(points, parameters, xy_horizontal_midline, u_new, v_new, p_new, parameters.output_file_prefix + "_xy_horizontal_midline.csv");
    vector<double> xy_vertical_midline;
    x = 0.0, y = -b_ellipse, nxy = 10 + (2 * b_ellipse / dxy);
    for (int i1 = 0; i1 < nxy; i1++)
    {
        if (fabs(y) <= b_ellipse)
            xy_vertical_midline.push_back(x), xy_vertical_midline.push_back(y);
        y = y + dxy;
    }
    write_interp_csv(points, parameters, xy_vertical_midline, u_new, v_new, p_new, parameters.output_file_prefix + "_xy_vertical_midline.csv");

    FILE *file;
    string output_file = parameters.output_file_prefix + "_steady_error.csv";
    file = fopen(output_file.c_str(), "w");
    fprintf(file, "time(s),l1_error\n");
    for (int it = 0; it < total_steady_l1_err_log.size(); it++)
        fprintf(file, "%.16g,%.16g\n", it * parameters.dt, total_steady_l1_err_log[it]);
    fclose(file);
    write_tecplot_temporal_variables(points, parameters, variable_names, variable_pointers, it + 1);
}