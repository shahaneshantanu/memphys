//Author: Dr. Shantanu Shahane
//compile: time make eccentric_couette_flow
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/navier_stokes.hpp"
#include "../../header_files/postprocessing_functions.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    PARAMETERS parameters("parameters_file.csv", "/home/shantanu/Desktop/All Simulation Results/Meshless_Methods/CAD_mesh_files/hole_geometries/eccentric_circle_in_circle/annulus_opencasc_n_30.msh");
    int temporal_order = 1;
    parameters.Courant = parameters.Courant / ((double)temporal_order); //Adam-Bashforth has half stability than explicit Euler

    double Re = 50.0, r_i = 1.0 / 3.0, u_theta_i = 0.5;
    parameters.rho = 10.0, parameters.mu = parameters.rho * u_theta_i * r_i / Re;

    POINTS points(parameters);
    CLOUD cloud(points, parameters);

    Eigen::VectorXd p_new = Eigen::VectorXd::Zero(points.nv + 1);
    Eigen::VectorXd u_new = Eigen::VectorXd::Zero(points.nv), v_new = u_new;
    Eigen::VectorXd u_old = u_new, v_old = v_new, p_old = p_new;
    double x, y, z, r;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    for (int iv = 0; iv < points.nv; iv++)
    {
        u_dirichlet_flag.push_back(true);  //initialize to dirichlet
        v_dirichlet_flag.push_back(true);  //initialize to dirichlet
        p_dirichlet_flag.push_back(false); //initialize to neumann
        x = points.xyz[parameters.dimension * iv], y = points.xyz[parameters.dimension * iv + 1], r = sqrt(x * x + y * y);
        if (points.boundary_flag[iv]) //inner boundary non-zero velocity BC
            if (fabs(r - r_i) < 1E-5)
                u_new[iv] = (-u_theta_i * y / r), v_new[iv] = (u_theta_i * x / r); //dirichlet BC
    }
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, max_abs(u_new), max_abs(v_new), 0.0, parameters.mu / parameters.rho);

    u_old = u_new, v_old = v_new;
    FRACTIONAL_STEP_1 fractional_step_1(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, temporal_order);
    double total_steady_err;
    clock_t clock_t1 = clock(), clock_t2 = clock();
    cout << "\nTime marching started\n\n";
    for (int it = 0; it < parameters.nt; it++)
    {
        total_steady_err = fractional_step_1.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, it);
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || total_steady_err < parameters.steady_tolerance) //|| true
        {
            printf("    pressure regularization alpha: %g\n", p_new[points.nv]);
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

    calc_navier_stokes_residuals_2D(points, parameters, u_new, v_new, p_new);
    parameters.total_timer = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    write_simulation_details(points, cloud, parameters), write_iteration_details(parameters);
    write_navier_stokes_residuals_2D(points, parameters, u_new, v_new, p_new, "_residuals_new.csv");
    write_navier_stokes_tecplot_2D(points, parameters, u_new, v_new, p_new);
}