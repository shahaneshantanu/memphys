//Author: Dr. Shantanu Shahane
//compile: time make bell_colella_JCP_periodic
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
    int nx = 64;
    string meshfile = "../../gmsh_files/Square/Square_n_" + to_string(nx) + "_unstruc.msh";
    PARAMETERS parameters("parameters_file.csv", meshfile);

    int dim = parameters.dimension, temporal_order = 2;
    double Re = 1E30;
    parameters.rho = 1.0, parameters.mu = parameters.rho / Re;
    double x, y, nu = parameters.mu / parameters.rho, t_end = 2.0, total_steady_l1_err, physical_time;
    POINTS points(parameters);
    vector<string> periodic_axis{"x", "y"};
    points.set_periodic_bc(parameters, periodic_axis);
    CLOUD cloud(points, parameters);
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, 1.0, 1.0, 0.0, parameters.mu / parameters.rho);

    int hypervisc_k = 2;
    double hypervisc_gamma = parameters.rho * pow(-1.0, 1.0 - hypervisc_k) * pow(2.0, -6.0) * pow(parameters.avg_dx, 2.0 * hypervisc_k - 1.0);
    cout << "\n\nhypervisc_gamma: " << hypervisc_gamma << "\n\n";

    // double dt_stable = parameters.dt;
    // parameters.dt = 0.00032;
    // parameters.Courant = parameters.Courant * parameters.dt / dt_stable;
    // cout << "Set dt: " << parameters.dt << ", Co: " << parameters.Courant << ", nt till t_end : " << t_end / parameters.dt << "\n\n ";

    Eigen::VectorXd p_new;
    p_new = Eigen::VectorXd::Zero(points.nv + 1);
    Eigen::VectorXd u_new = Eigen::VectorXd::Zero(points.nv), v_new = u_new, vorticity = u_new;
    Eigen::VectorXd u_old = u_new, v_old = v_new, p_old = p_new;
    Eigen::VectorXd xmom_source = u_new, ymom_source = v_new;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    for (int iv = 0; iv < points.nv; iv++)
    {
        u_dirichlet_flag.push_back(true);  //initialize to dirichlet
        v_dirichlet_flag.push_back(true);  //initialize to dirichlet
        p_dirichlet_flag.push_back(false); //initialize to neumann
        x = points.xyz[parameters.dimension * iv], y = points.xyz[parameters.dimension * iv + 1];
        if (y <= 0.5)
            u_new[iv] = tanh(30.0 * (y - 0.25));
        else
            u_new[iv] = tanh(30.0 * (0.75 - y));
        v_new[iv] = 0.05 * sin(2.0 * M_PI * x);
        // u_new[iv] = exp(-(y - 0.5) * (y - 0.5) / (2 * 0.1 * 0.1));
        // v_new[iv] = 0.05 * sin(2.0 * M_PI * x);
    }
    u_old = u_new, v_old = v_new;
    FRACTIONAL_STEP_1 fractional_step_1(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, temporal_order);
    vorticity = (points.grad_x_matrix_EIGEN * v_new) - (points.grad_y_matrix_EIGEN * u_new);
    vector<string> variable_names{"u_new", "v_new", "p_new", "vorticity"};
    vector<Eigen::VectorXd *> variable_pointers{&u_new, &v_new, &p_new, &vorticity};
    write_tecplot_temporal_variables_header(points, parameters, variable_names);
    write_tecplot_temporal_variables(points, parameters, variable_names, variable_pointers, 0);

    //composite 2D integration logic: http://www.cas.mcmaster.ca/~qiao//courses/cs4xo3/slides/ch04.pdf
    Eigen::VectorXd integrate_wt = Eigen::VectorXd::Zero(nx + 1), interp_vel_square = Eigen::VectorXd::Zero((nx + 1) * (nx + 1)), energy_temp, u_interp = interp_vel_square, v_interp = interp_vel_square;
    Eigen::MatrixXd interp_vel_square_2d = Eigen::MatrixXd::Zero(nx + 1, nx + 1);
    for (int i1 = 0; i1 <= nx; i1++)
    {
        if (i1 % 2 == 0)
            integrate_wt(i1) = 2.0 / 3.0;
        else if (i1 % 2 == 1)
            integrate_wt(i1) = 4.0 / 3.0;
        if (i1 == 0 || i1 == nx)
            integrate_wt(i1) = 1.0 / 3.0;
    }
    // cout << "\n\nintegrate_wt: " << integrate_wt.transpose() << "\n\n";
    double interp_dx = 1.0 / (nx);
    vector<double> interp_xy, energy;
    for (int ix = 0; ix <= nx; ix++)
        for (int iy = 0; iy <= nx; iy++)
            interp_xy.push_back(ix * interp_dx), interp_xy.push_back(iy * interp_dx);
    Eigen::SparseMatrix<double, Eigen::RowMajor> interp_matrix = calc_interp_matrix(interp_xy, points, parameters);

    u_interp = interp_matrix * u_new, v_interp = interp_matrix * v_new;
    interp_vel_square = u_interp.cwiseProduct(u_interp) + v_interp.cwiseProduct(v_interp);
    // interp_vel_square = interp_matrix * (u_new.cwiseProduct(u_new) + v_new.cwiseProduct(v_new));
    for (int ix = 0; ix <= nx; ix++)
        for (int iy = 0; iy <= nx; iy++)
            interp_vel_square_2d(ix, iy) = interp_vel_square(ix * (nx + 1) + iy);
    energy_temp = (interp_dx * interp_dx) * (integrate_wt.transpose() * interp_vel_square_2d * integrate_wt);
    energy.push_back(energy_temp(0));

    clock_t clock_t1 = clock(), clock_t2 = clock();
    cout << "\nTime marching started\n\n";
    for (int it = 0; it < parameters.nt; it++)
    {
        xmom_source = hypervisc_gamma * u_old, ymom_source = hypervisc_gamma * v_old;
        for (int i1 = 0; i1 < hypervisc_k; i1++)
        {
            xmom_source = points.laplacian_matrix_EIGEN * xmom_source;
            ymom_source = points.laplacian_matrix_EIGEN * ymom_source;
        }

        physical_time = (it + 1.0) * parameters.dt;
        fractional_step_1.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, xmom_source, ymom_source, it);
        total_steady_l1_err = (u_new - u_old).lpNorm<1>() / (u_new.lpNorm<Eigen::Infinity>());
        total_steady_l1_err += ((v_new - v_old).lpNorm<1>() / (v_new.lpNorm<Eigen::Infinity>()));
        total_steady_l1_err = total_steady_l1_err / (parameters.dimension * parameters.dt * u_new.size());
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;

        u_interp = interp_matrix * u_new, v_interp = interp_matrix * v_new;
        interp_vel_square = u_interp.cwiseProduct(u_interp) + v_interp.cwiseProduct(v_interp);
        // interp_vel_square = interp_matrix * (u_new.cwiseProduct(u_new) + v_new.cwiseProduct(v_new));
        for (int ix = 0; ix <= nx; ix++)
            for (int iy = 0; iy <= nx; iy++)
                interp_vel_square_2d(ix, iy) = interp_vel_square(ix * (nx + 1) + iy);
        energy_temp = (interp_dx * interp_dx) * (integrate_wt.transpose() * interp_vel_square_2d * integrate_wt);
        energy.push_back(energy_temp(0));

        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || ((it + 1.0) * parameters.dt >= t_end)) //|| true
        {
            printf("    pressure regularization alpha: %g, energy: %g\n", p_new[points.nv], energy_temp(0));
            printf("    total steady state l1_error: %g, steady_tolerance: %g\n", total_steady_l1_err, parameters.steady_tolerance);
            printf("    Completed it: %i, dt: %g, physical time: %g, in CPU time: %g seconds\n\n", it, parameters.dt, physical_time, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
        parameters.nt_actual = it + 1;
        u_old = u_new, v_old = v_new, p_old = p_new;
        if ((it + 1.0) * parameters.dt >= t_end || (it == parameters.nt - 1) || (fabs((int)(5.0 * physical_time) - (5.0 * physical_time)) <= (5.0 * parameters.dt)))
        {
            vorticity = (points.grad_x_matrix_EIGEN * v_new) - (points.grad_y_matrix_EIGEN * u_new);
            write_tecplot_temporal_variables(points, parameters, variable_names, variable_pointers, it + 1);
        }
        if ((it + 1.0) * parameters.dt >= t_end)
            break;
    }
    parameters.solve_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    printf("Time marching ended; factoring_timer: %g, solve_timer:%g seconds\n\n", parameters.factoring_timer, parameters.solve_timer);
    write_simulation_details(points, cloud, parameters);

    FILE *file;
    string output_file = parameters.output_file_prefix + "_energy.csv";
    file = fopen(output_file.c_str(), "w");
    fprintf(file, "time(s),l1_error\n");
    for (int it = 0; it < energy.size(); it++)
        fprintf(file, "%.16g,%.16g\n", it * parameters.dt, energy[it]);
    fclose(file);
}
