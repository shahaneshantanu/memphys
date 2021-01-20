//Author: Dr. Shantanu Shahane
//compile: time make taylor_green_modified
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/navier_stokes.hpp"
#include "../../header_files/postprocessing_functions.hpp"
#include "../../header_files/coefficient_computations.hpp"
using namespace std;

void calc_errors(Eigen::VectorXd &ana_val, Eigen::VectorXd &num_val, vector<double> &max_err, vector<double> &l1_err, vector<double> &max_err_boundary, vector<double> &l1_err_boundary, vector<double> &max_err_internal, vector<double> &l1_err_internal, POINTS &points)
{
    double max_err_boundary1, l1_err_boundary1, max_err_internal1, l1_err_internal1, max_err1, l1_err1;
    calc_max_l1_error(ana_val, num_val, max_err_boundary1, l1_err_boundary1, max_err_internal1, l1_err_internal1, points.boundary_flag);
    calc_max_l1_error(ana_val, num_val, max_err1, l1_err1);
    max_err.push_back(max_err1), l1_err.push_back(l1_err1);
    max_err_boundary.push_back(max_err_boundary1), l1_err_boundary.push_back(l1_err_boundary1);
    max_err_internal.push_back(max_err_internal1), l1_err_internal.push_back(l1_err_internal1);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    PARAMETERS parameters("parameters_file.csv", "/home/shantanu/Desktop/All Simulation Results/Meshless_Methods/CAD_mesh_files/Square/gmsh/Square_n_40_unstruc.msh");

    int dim = parameters.dimension, temporal_order = 2;
    parameters.Courant = parameters.Courant / ((double)temporal_order); //Adam-Bashforth has half stability than explicit Euler
    double Re = 100.0, t_end = 1.0;
    parameters.rho = 10.0, parameters.mu = parameters.rho / Re;
    POINTS points(parameters);
    CLOUD cloud(points, parameters);
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, 1.0, 1.0, 0.0, parameters.mu / parameters.rho);

    Eigen::VectorXd p_ana, p_new, p_ana_head, p_new_head;
    p_new = Eigen::VectorXd::Zero(points.nv + 1), p_ana = p_new;
    Eigen::VectorXd u_new = Eigen::VectorXd::Zero(points.nv), v_new = u_new;
    Eigen::VectorXd u_ana = u_new, v_ana = v_new;
    Eigen::VectorXd x_mom_source = u_new, y_mom_source = v_new;
    Eigen::VectorXd u_old = u_new, v_old = v_new, p_old = p_new;
    double x, y, F_t, nu = parameters.mu / parameters.rho;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    for (int iv = 0; iv < points.nv; iv++)
        u_dirichlet_flag.push_back(true), v_dirichlet_flag.push_back(true), p_dirichlet_flag.push_back(false);

    vector<double> u_max_err_internal, u_l1_err_internal, u_max_err_boundary, u_l1_err_boundary, u_max_err, u_l1_err;
    vector<double> v_max_err_internal, v_l1_err_internal, v_max_err_boundary, v_l1_err_boundary, v_max_err, v_l1_err;
    vector<double> p_max_err_internal, p_l1_err_internal, p_max_err_boundary, p_l1_err_boundary, p_max_err, p_l1_err;

    FRACTIONAL_STEP_1 fractional_step_1(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, temporal_order);
    clock_t clock_t1 = clock(), clock_t2 = clock();
    cout << "\nTime marching started\n\n";
    parameters.nt = ceil(t_end / parameters.dt);
    F_t = sin(0 * parameters.dt * M_PI);
    for (int iv = 0; iv < points.nv; iv++)
    {
        x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1];
        u_ana[iv] = cos(x) * sin(y) * F_t;
        v_ana[iv] = -sin(x) * cos(y) * F_t;
        p_ana[iv] = -0.25 * parameters.rho * (cos(2 * x) + cos(2 * y)) * F_t * F_t;
        u_old[iv] = u_ana[iv], v_old[iv] = v_ana[iv]; //initial condition
    }
    for (int it = 0; it < parameters.nt; it++)
    {
        F_t = sin((it + 1) * parameters.dt * M_PI);
        for (int iv = 0; iv < points.nv; iv++)
        {
            x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1];
            u_ana[iv] = cos(x) * sin(y) * F_t;
            v_ana[iv] = -sin(x) * cos(y) * F_t;
            p_ana[iv] = -0.25 * parameters.rho * (cos(2 * x) + cos(2 * y)) * sin((it + 0.5) * parameters.dt * M_PI) * sin((it + 0.5) * parameters.dt * M_PI);
            x_mom_source[iv] = parameters.rho * cos(x) * sin(y) * (M_PI * cos(it * parameters.dt * M_PI) + 2 * nu * sin(it * parameters.dt * M_PI));
            y_mom_source[iv] = -parameters.rho * sin(x) * cos(y) * (M_PI * cos(it * parameters.dt * M_PI) + 2 * nu * sin(it * parameters.dt * M_PI));
            if (points.boundary_flag[iv])
                u_old[iv] = u_ana[iv], v_old[iv] = v_ana[iv]; //dirichlet BC
        }

        fractional_step_1.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, x_mom_source, y_mom_source, it);
        u_old = u_new, v_old = v_new, p_old = p_new;

        p_ana_head = p_ana.head(points.nv), p_new_head = p_new.head(points.nv);
        p_new_head = p_new_head - (Eigen::VectorXd::Ones(points.nv) * (p_new_head[0] - p_ana_head[0])); //reset level to analytical solution
        calc_errors(u_ana, u_new, u_max_err, u_l1_err, u_max_err_boundary, u_l1_err_boundary, u_max_err_internal, u_l1_err_internal, points);
        calc_errors(v_ana, v_new, v_max_err, v_l1_err, v_max_err_boundary, v_l1_err_boundary, v_max_err_internal, v_l1_err_internal, points);
        calc_errors(p_ana_head, p_new_head, p_max_err, p_l1_err, p_max_err_boundary, p_l1_err_boundary, p_max_err_internal, p_l1_err_internal, points);

        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1) //|| true
        {
            printf("    X-vel: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", u_max_err_internal[it], u_l1_err_internal[it], u_max_err_boundary[it], u_l1_err_boundary[it], u_max_err[it], u_l1_err[it]);
            printf("    Y-vel: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", v_max_err_internal[it], v_l1_err_internal[it], v_max_err_boundary[it], v_l1_err_boundary[it], v_max_err[it], v_l1_err[it]);
            printf("    Pressure: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", p_max_err_internal[it], p_l1_err_internal[it], p_max_err_boundary[it], p_l1_err_boundary[it], p_max_err[it], p_l1_err[it]);

            printf("    pressure regularization alpha: %g\n", p_new[points.nv]);
            printf("    Completed it: %i of nt: %i (%.4g percent), dt: %g, in CPU time: %g seconds\n\n", it, parameters.nt, (100.0 * (it + 1)) / parameters.nt, parameters.dt, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
    }
    parameters.solve_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    printf("Time marching ended; factoring_timer: %g, solve_timer:%g seconds\n\n", parameters.factoring_timer, parameters.solve_timer);

    cout << "\nError statistics over time (max., avg.)\n";
    printf("X-vel: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", *max_element(u_max_err_internal.begin(), u_max_err_internal.end()), accumulate(u_l1_err_internal.begin(), u_l1_err_internal.end(), 0.0) / u_l1_err_internal.size(), *max_element(u_max_err_boundary.begin(), u_max_err_boundary.end()), accumulate(u_l1_err_boundary.begin(), u_l1_err_boundary.end(), 0.0) / u_l1_err_boundary.size(), *max_element(u_max_err.begin(), u_max_err.end()), accumulate(u_l1_err.begin(), u_l1_err.end(), 0.0) / u_l1_err.size());
    printf("Y-vel: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", *max_element(v_max_err_internal.begin(), v_max_err_internal.end()), accumulate(v_l1_err_internal.begin(), v_l1_err_internal.end(), 0.0) / v_l1_err_internal.size(), *max_element(v_max_err_boundary.begin(), v_max_err_boundary.end()), accumulate(v_l1_err_boundary.begin(), v_l1_err_boundary.end(), 0.0) / v_l1_err_boundary.size(), *max_element(v_max_err.begin(), v_max_err.end()), accumulate(v_l1_err.begin(), v_l1_err.end(), 0.0) / v_l1_err.size());
    printf("Pressure: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", *max_element(p_max_err_internal.begin(), p_max_err_internal.end()), accumulate(p_l1_err_internal.begin(), p_l1_err_internal.end(), 0.0) / p_l1_err_internal.size(), *max_element(p_max_err_boundary.begin(), p_max_err_boundary.end()), accumulate(p_l1_err_boundary.begin(), p_l1_err_boundary.end(), 0.0) / p_l1_err_boundary.size(), *max_element(p_max_err.begin(), p_max_err.end()), accumulate(p_l1_err.begin(), p_l1_err.end(), 0.0) / p_l1_err.size());

    FILE *file;
    string file_name = parameters.output_file_prefix + "_temporal_errors.csv";
    file = fopen(file_name.c_str(), "w");
    fprintf(file, "it,time");
    fprintf(file, ",,u_max_err_internal, u_l1_err_internal, u_max_err_boundary, u_l1_err_boundary, u_max_err, u_l1_err");
    fprintf(file, ",,v_max_err_internal, v_l1_err_internal, v_max_err_boundary, v_l1_err_boundary, v_max_err, v_l1_err");
    fprintf(file, ",,p_max_err_internal, p_l1_err_internal, p_max_err_boundary, p_l1_err_boundary, p_max_err, p_l1_err");
    fprintf(file, "\n");
    for (int it = 0; it < parameters.nt; it++)
    {
        fprintf(file, "%i,%.16g", it, it * parameters.dt);
        fprintf(file, ",,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g", u_max_err_internal[it], u_l1_err_internal[it], u_max_err_boundary[it], u_l1_err_boundary[it], u_max_err[it], u_l1_err[it]);
        fprintf(file, ",,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g", v_max_err_internal[it], v_l1_err_internal[it], v_max_err_boundary[it], v_l1_err_boundary[it], v_max_err[it], v_l1_err[it]);
        fprintf(file, ",,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g", p_max_err_internal[it], p_l1_err_internal[it], p_max_err_boundary[it], p_l1_err_boundary[it], p_max_err[it], p_l1_err[it]);
        fprintf(file, "\n ");
    }
    fclose(file);
    file_name = parameters.output_file_prefix + "_temporal_stats.csv";
    file = fopen(file_name.c_str(), "w");
    fprintf(file, "u_max_err_internal,%.16g \nu_l1_err_internal,%.16g \nu_max_err_boundary,%.16g \nu_l1_err_boundary,%.16g \nu_max_err,%.16g \nu_l1_err,%.16g\n", *max_element(u_max_err_internal.begin(), u_max_err_internal.end()), accumulate(u_l1_err_internal.begin(), u_l1_err_internal.end(), 0.0) / u_l1_err_internal.size(), *max_element(u_max_err_boundary.begin(), u_max_err_boundary.end()), accumulate(u_l1_err_boundary.begin(), u_l1_err_boundary.end(), 0.0) / u_l1_err_boundary.size(), *max_element(u_max_err.begin(), u_max_err.end()), accumulate(u_l1_err.begin(), u_l1_err.end(), 0.0) / u_l1_err.size());
    fprintf(file, "v_max_err_internal,%.16g \nv_l1_err_internal,%.16g \nv_max_err_boundary,%.16g \nv_l1_err_boundary,%.16g \nv_max_err,%.16g \nv_l1_err,%.16g\n", *max_element(v_max_err_internal.begin(), v_max_err_internal.end()), accumulate(v_l1_err_internal.begin(), v_l1_err_internal.end(), 0.0) / v_l1_err_internal.size(), *max_element(v_max_err_boundary.begin(), v_max_err_boundary.end()), accumulate(v_l1_err_boundary.begin(), v_l1_err_boundary.end(), 0.0) / v_l1_err_boundary.size(), *max_element(v_max_err.begin(), v_max_err.end()), accumulate(v_l1_err.begin(), v_l1_err.end(), 0.0) / v_l1_err.size());
    fprintf(file, "p_max_err_internal,%.16g \np_l1_err_internal,%.16g \np_max_err_boundary,%.16g \np_l1_err_boundary,%.16g \np_max_err,%.16g \np_l1_err,%.16g\n", *max_element(p_max_err_internal.begin(), p_max_err_internal.end()), accumulate(p_l1_err_internal.begin(), p_l1_err_internal.end(), 0.0) / p_l1_err_internal.size(), *max_element(p_max_err_boundary.begin(), p_max_err_boundary.end()), accumulate(p_l1_err_boundary.begin(), p_l1_err_boundary.end(), 0.0) / p_l1_err_boundary.size(), *max_element(p_max_err.begin(), p_max_err.end()), accumulate(p_l1_err.begin(), p_l1_err.end(), 0.0) / p_l1_err.size());
    fclose(file);
    parameters.total_timer = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    write_simulation_details(points, cloud, parameters), write_iteration_details(parameters);
}