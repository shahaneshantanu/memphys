//Author: Dr. Shantanu Shahane
//compile: time make kovasznay_flow
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/navier_stokes.hpp"
#include "../../header_files/postprocessing_functions.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    PARAMETERS parameters("parameters_file.csv", "/home/shantanu/Desktop/All Simulation Results/Meshless_Methods/CAD_mesh_files/Square/gmsh/Square_n_20_unstruc.msh");
    int temporal_order = 1;
    parameters.Courant = parameters.Courant / ((double)temporal_order); //Adam-Bashforth has half stability than explicit Euler

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

    calc_navier_stokes_errors_2D(points, parameters, u_ana, v_ana, p_ana, u_new, v_new, p_new);
    calc_navier_stokes_residuals_2D(points, parameters, u_ana, v_ana, p_ana);
    printf("main: Re: %g, lambda: %g\n\n", Re, lambda);
    parameters.total_timer = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    write_simulation_details(points, cloud, parameters), write_iteration_details(parameters);
    write_navier_stokes_errors_2D(points, parameters, u_ana, v_ana, p_ana, u_new, v_new, p_new);
    write_navier_stokes_residuals_2D(points, parameters, u_ana, v_ana, p_ana, "_residuals_ana.csv");
    write_navier_stokes_residuals_2D(points, parameters, u_new, v_new, p_new, "_residuals_new.csv");
    write_navier_stokes_tecplot_2D(points, parameters, u_ana, v_ana, p_ana, u_new, v_new, p_new);
}