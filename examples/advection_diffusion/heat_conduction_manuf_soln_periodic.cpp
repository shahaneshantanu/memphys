//Author: Dr. Shantanu Shahane
//compile: time make heat_conduction_manuf_soln_periodic
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    double wv = 2.0 * M_PI; //wavenumber

    string meshfile = "/media/shantanu/Data/All Simulation Results/Meshless_Methods/CAD_mesh_files/Square/gmsh/Square_n_40_unstruc.msh"; //2D example
    // string meshfile = "/media/shantanu/Data/All Simulation Results/Meshless_Methods/CAD_mesh_files/cuboid/Cuboid_n_40_unstruc.msh"; //3D example

    PARAMETERS parameters("parameters_file.csv", meshfile);
    POINTS points(parameters);
    vector<string> periodic_axis{"y", "x"};
    points.set_periodic_bc(parameters, periodic_axis);
    CLOUD cloud(points, parameters);

    Eigen::VectorXd T_num_new;
    vector<bool> dirichlet_flag;
    for (int iv = 0; iv < points.nv; iv++)
        dirichlet_flag.push_back(true);
    T_num_new = Eigen::VectorXd::Zero(points.nv + 1); //initialize assuming full Neumann
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv] && dirichlet_flag[iv])
        { //identified at least one boundary point with dirichlet BC (no regularization needed)
            T_num_new = Eigen::VectorXd::Zero(points.nv);
            break;
        }
    Eigen::VectorXd T_ana = T_num_new, T_num_old = T_num_new, source = T_num_new;
    double max_err, l1_err, grad_x, grad_y, grad_z = 0.0, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal;
    int dim = parameters.dimension;
    for (int iv = 0; iv < points.nv; iv++)
    { //manufactured solution: sin(wv*x)*sin(wv*y)*sin(wv*z)
        T_ana[iv] = sin(wv * points.xyz[dim * iv]) * sin(wv * points.xyz[dim * iv + 1]);
        if (dim == 3)
            T_ana[iv] = T_ana[iv] * sin(wv * points.xyz[dim * iv + 2]);
    }
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, 0.0, 0.0, 0.0, 1.0);
    clock_t t1 = clock();
    double unsteady_term_coeff = 1.0 / parameters.dt, conv_term_coeff = 0.0, diff_term_coeff = -1.0;
    SOLVER solver_T;
    solver_T.init(points, cloud, parameters, dirichlet_flag, unsteady_term_coeff, conv_term_coeff, diff_term_coeff, true);
    parameters.factoring_timer = ((double)(clock() - t1)) / CLOCKS_PER_SEC;
    clock_t clock_t1 = clock(), clock_t2 = clock();
    for (int it = 0; it < parameters.nt; it++)
    {
        for (int iv = 0; iv < points.nv; iv++)
        {
            if (points.boundary_flag[iv])
            {
                if (dirichlet_flag[iv])
                    source[iv] = T_ana[iv]; //dirichlet BC
                else
                {
                    grad_x = wv * cos(wv * points.xyz[dim * iv]) * sin(wv * points.xyz[dim * iv + 1]);
                    grad_y = wv * sin(wv * points.xyz[dim * iv]) * cos(wv * points.xyz[dim * iv + 1]);
                    if (dim == 3)
                    {
                        grad_x = grad_x * sin(wv * points.xyz[dim * iv + 2]);
                        grad_y = grad_y * sin(wv * points.xyz[dim * iv + 2]);
                        grad_z = points.normal[dim * iv + 2] * wv * sin(wv * points.xyz[dim * iv]) * sin(wv * points.xyz[dim * iv + 1]) * cos(wv * points.xyz[dim * iv + 2]);
                    }
                    grad_x = grad_x * points.normal[dim * iv];
                    grad_y = grad_y * points.normal[dim * iv + 1];
                    source[iv] = grad_x + grad_y + grad_z;
                }
            }
            else
            {
                source[iv] = -diff_term_coeff * wv * wv * dim * T_ana[iv];       //manufactured solution source term
                source[iv] = source[iv] + (T_num_old[iv] * unsteady_term_coeff); //unsteady term
            }
        }
        solver_T.general_solve(points, parameters, T_num_new, T_num_old, source);
        calc_max_l1_relative_error(T_num_old, T_num_new, max_err, l1_err);
        l1_err = l1_err / parameters.dt, max_err = max_err / parameters.dt;
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || l1_err < parameters.steady_tolerance)
        {
            printf("    main total steady state error: %g, steady_tolerance: %g\n", l1_err, parameters.steady_tolerance);
            printf("    main Completed it: %i of nt: %i, dt: %g, in CPU time: %.2g seconds\n\n", it, parameters.nt, parameters.dt, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
        parameters.steady_error_log.push_back(l1_err);
        if (l1_err < parameters.steady_tolerance && it > 1)
            break;
        T_num_old = T_num_new;
    }
    parameters.solve_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    if (T_num_new.rows() == points.nv + 1) //reset level to analytical solution
    {
        T_num_new = T_num_new - (Eigen::VectorXd::Ones(points.nv + 1) * (T_num_new[0] - T_ana[0]));
        T_num_new[points.nv] = T_ana[points.nv];
    }
    calc_max_l1_relative_error(T_ana, T_num_new, max_err, l1_err);
    printf("\nmain dimension: %i, relative errors in T: max: %g, avg: %g\n", parameters.dimension, max_err, l1_err);
    parameters.total_timer = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    write_simulation_details(points, cloud, parameters), write_iteration_details(parameters);
    Eigen::VectorXd T_error = (T_ana - T_num_new).cwiseAbs();
    vector<string> variable_names{"T_new", "T_ana", "T_error"};
    vector<Eigen::VectorXd *> variable_pointers{&T_num_new, &T_ana, &T_error};
    write_tecplot_steady_variables(points, parameters, variable_names, variable_pointers);

    FILE *file;
    string filename = parameters.output_file_prefix + "_error.csv";
    file = fopen(filename.c_str(), "w");
    fprintf(file, ",max_internal,avg_internal,max_boundary,avg_boundary,max_overall,avg_overall\n");
    calc_max_l1_relative_error(T_ana, T_num_new, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_relative_error(T_ana, T_num_new, max_err, l1_err);
    fprintf(file, "Relative_Error,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);
    fclose(file);
}