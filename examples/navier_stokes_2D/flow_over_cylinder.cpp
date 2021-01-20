//Author: Dr. Shantanu Shahane
//compile: time make flow_over_cylinder
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/navier_stokes.hpp"
#include "../../header_files/postprocessing_functions.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    PARAMETERS parameters("parameters_file.csv", "/home/shantanu/Desktop/All Simulation Results/Meshless_Methods/CAD_mesh_files/hole_geometries/circle_in_rectangle/short domain/circle_in_rectangle_n_30.msh");
    int dim = parameters.dimension, temporal_order = 2;
    parameters.Courant = parameters.Courant / ((double)temporal_order); //Adam-Bashforth has half stability than explicit Euler

    POINTS points(parameters);
    CLOUD cloud(points, parameters);
    Eigen::VectorXd p_new = Eigen::VectorXd::Zero(points.nv);
    Eigen::VectorXd u_new = Eigen::VectorXd::Zero(points.nv), v_new = u_new;
    Eigen::VectorXd u_old = u_new, v_old = v_new, p_old = p_new;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    double x, y, r;

    /////////// Modify this section
    double Re = 20.0, radius = 1.0, u_bulk = 1.0;
    parameters.rho = 10.0, parameters.mu = parameters.rho * u_bulk * 2.0 * radius / Re;
    int tecplot_it_frequency = 100; //tecplot file written after every tecplot_it_frequency timesteps
    for (int iv = 0; iv < points.nv; iv++)
    { // Set boundary conditions here
        x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1], r = sqrt(x * x + y * y);
        u_dirichlet_flag.push_back(true); //initialize to dirichlet
        v_dirichlet_flag.push_back(true); //initialize to dirichlet
        p_dirichlet_flag.push_back(true); //initialize to dirichlet
        u_new[iv] = u_bulk;               //initialize entire domain to bulk velocity
        if (points.boundary_flag[iv])
        {
            if (fabs(r - radius) < 1E-5) //dirichlet BC along cylinder
                u_new[iv] = 0.0, v_new[iv] = 0.0, u_dirichlet_flag[iv] = true, v_dirichlet_flag[iv] = true, p_dirichlet_flag[iv] = false;
            if (fabs(x + 10.0) < 1E-5) //dirichlet BC at inlet
                u_new[iv] = u_bulk, v_new[iv] = 0.0, u_dirichlet_flag[iv] = true, v_dirichlet_flag[iv] = true, p_dirichlet_flag[iv] = false;
            if (fabs(x - 20.0) < 1E-5) //outlet BC
                p_dirichlet_flag[iv] = true;
            if ((fabs(y - 15.0) < 1E-5) || (fabs(y + 15.0) < 1E-5))
            {                                                 //symmetry: zero shear stress
                u_dirichlet_flag[iv] = false;                 //neumann at top and bottom
                v_new[iv] = 0.0, v_dirichlet_flag[iv] = true; //dirichlet at top and bottom
                p_dirichlet_flag[iv] = false;
            }
        }
    }
    u_old = u_new, v_old = v_new;
    ///////////

    points.calc_elem_bc_tag(parameters);
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, max_abs(u_new), max_abs(v_new), 0.0, parameters.mu / parameters.rho);

    FRACTIONAL_STEP_1 fractional_step_1(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, temporal_order);
    double total_steady_err;
    if (tecplot_it_frequency > 0)
    {
        write_navier_stokes_tecplot_temporal_header(points, parameters);
        write_navier_stokes_tecplot_temporal_fields(points, parameters, u_new, v_new, u_new, p_new, 0);
    }
    clock_t clock_t1 = clock(), clock_t2 = clock();
    cout << "\nTime marching started\n\n";
    for (int it = 0; it < parameters.nt; it++)
    {
        total_steady_err = fractional_step_1.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, it);
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || total_steady_err < parameters.steady_tolerance) //|| true
        {
            // printf("    pressure regularization alpha: %g\n", p_new[points.nv]);
            printf("    total steady state error: %g, steady_tolerance: %g\n", total_steady_err, parameters.steady_tolerance);
            printf("    Completed it: %i of nt: %i, dt: %g, in CPU time: %g seconds\n\n", it, parameters.nt, parameters.dt, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
        if (tecplot_it_frequency > 0)
            if (it % tecplot_it_frequency == 0 || total_steady_err < parameters.steady_tolerance || it == parameters.nt - 1)
                write_navier_stokes_tecplot_temporal_fields(points, parameters, u_new, v_new, u_new, p_new, it + 1);
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
}