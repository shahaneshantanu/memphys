//Author: Dr. Shantanu Shahane
//compile: time make ab2_unsteady_heat_cond_manuf_sol
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();

    string meshfile = "../../gmsh_files/Square/Square_n_40_unstruc.msh";
    double alpha = 0.1, max_err, l1_err, x, y, t_end = 1.0;
    vector<double> max_err_t, l1_err_t;

    PARAMETERS parameters("parameters_file.csv", meshfile);
    POINTS points(parameters);
    CLOUD cloud(points, parameters);
    Eigen::VectorXd T_new = Eigen::VectorXd::Zero(points.nv), T_old = Eigen::VectorXd::Zero(points.nv), T_old_old = Eigen::VectorXd::Zero(points.nv), T_ana = Eigen::VectorXd::Zero(points.nv), T_source_old = Eigen::VectorXd::Zero(points.nv), T_source_old_old = Eigen::VectorXd::Zero(points.nv);
    vector<bool> dirichlet_flag;
    for (int iv = 0; iv < points.nv; iv++)
        dirichlet_flag.push_back(true);
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, 0.0, 0.0, 0.0, alpha);
    parameters.dt = 0.5 * parameters.dt; //Adam-Bashforth has half stability than explicit Euler
    parameters.nt = ceil(t_end / parameters.dt);
    int dim = parameters.dimension;
    clock_t clock_t1 = clock(), clock_t2 = clock();
    for (int it = 0; it < parameters.nt; it++)
    {
        for (int iv = 0; iv < points.nv; iv++)
        {
            x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1];
            T_ana[iv] = sin(M_PI * parameters.dt * (it + 1.0)) * sin(x) * sin(y);
            T_source_old[iv] = dim * alpha * sin(M_PI * parameters.dt * it) * sin(x) * sin(y);
            T_source_old[iv] = T_source_old[iv] + M_PI * cos(M_PI * parameters.dt * it) * sin(x) * sin(y);
            if (it > 0)
            {
                T_source_old_old[iv] = dim * alpha * sin(M_PI * parameters.dt * (it - 1.0)) * sin(x) * sin(y);
                T_source_old_old[iv] = T_source_old_old[iv] + M_PI * cos(M_PI * parameters.dt * (it - 1.0)) * sin(x) * sin(y);
            }
        }
        if (it == 0)
        { //forward euler first time
            T_old = T_old_old + ((alpha * parameters.dt) * (points.laplacian_matrix_EIGEN_internal * T_old_old)) + (parameters.dt * T_source_old);
            for (int iv = 0; iv < points.nv; iv++)
                if (points.boundary_flag[iv])
                    T_old[iv] = T_ana[iv]; //dirichlet BC
        }
        else
        { //Adam=Bashforth subsequently
            T_new = T_old + ((alpha * parameters.dt) * (points.laplacian_matrix_EIGEN_internal * (1.5 * T_old - 0.5 * T_old_old))) + (parameters.dt * (1.5 * T_source_old - 0.5 * T_source_old_old));
            for (int iv = 0; iv < points.nv; iv++)
                if (points.boundary_flag[iv])
                    T_new[iv] = T_ana[iv]; //dirichlet BC
        }
        if (it == 0)
            calc_max_l1_error(T_ana, T_old, max_err, l1_err);
        else
            calc_max_l1_error(T_ana, T_new, max_err, l1_err);
        max_err_t.push_back(max_err), l1_err_t.push_back(l1_err);
        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1)
            printf("    main Completed it: %i of nt: %i (%.2f percent), dt: %g, in CPU time: %.2g seconds\n\n", it + 1, parameters.nt, 100.0 * (it + 1.0) / parameters.nt, parameters.dt, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC), clock_t2 = clock();
        if (it > 0)
            T_old_old = T_old, T_old = T_new;
    }

    printf("\nError statistics over time (max., avg.): (%g, %g)\n", *max_element(max_err_t.begin(), max_err_t.end()), accumulate(l1_err_t.begin(), l1_err_t.end(), 0.0) / l1_err_t.size());

    FILE *file;
    string file_name = parameters.output_file_prefix + "_ab2_temporal_errors.csv";
    file = fopen(file_name.c_str(), "w");
    fprintf(file, "it,time,max_err,l1_err\n");
    for (int it = 0; it < parameters.nt; it++)
        fprintf(file, "%i,%.16g,%.16g,%.16g\n", it, it * parameters.dt, max_err_t[it], l1_err_t[it]);
    fclose(file);
}