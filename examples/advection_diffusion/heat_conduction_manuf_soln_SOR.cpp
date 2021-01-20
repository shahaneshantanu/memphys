//Author: Dr. Shantanu Shahane
//compile: time make heat_conduction_manuf_soln_SOR
//execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
using namespace std;

void assemble_matrix(Eigen::SparseMatrix<double, Eigen::ColMajor> &matrix_eigen, POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &dirichlet_flag, double diff_term_coeff, int system_size)
{
    vector<Eigen::Triplet<double>> triplet;
    int dim = parameters.dimension, iv_nb;
    double val;
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (points.boundary_flag[iv])
        { //lies on boundary
            if (dirichlet_flag[iv])
            { //apply dirichlet BC
                triplet.push_back(Eigen::Triplet<double>(iv, iv, 1.0));
            }
            else
            { //apply neumann BC
                for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                {
                    iv_nb = cloud.nb_points_col[i1];
                    val = points.normal[dim * iv] * cloud.grad_x_coeff[i1] + points.normal[dim * iv + 1] * cloud.grad_y_coeff[i1];
                    if (dim == 3)
                        val += points.normal[dim * iv + 2] * cloud.grad_z_coeff[i1];
                    triplet.push_back(Eigen::Triplet<double>(iv, iv_nb, val));
                }
            }
        }
        else
        {
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                iv_nb = cloud.nb_points_col[i1];
                val = diff_term_coeff * cloud.laplacian_coeff[i1]; //diffusion
                triplet.push_back(Eigen::Triplet<double>(iv, iv_nb, val));
            }
        }
    }
    if (system_size == points.nv + 1)
    { //extra constraint (regularization: http://www-e6.ijs.si/medusa/wiki/index.php/Poisson%27s_equation) for neumann BCs
        for (int iv = 0; iv < points.nv; iv++)
        {
            triplet.push_back(Eigen::Triplet<double>(iv, points.nv, 1.0)); //last column
            triplet.push_back(Eigen::Triplet<double>(points.nv, iv, 1.0)); //last row
        }
        triplet.push_back(Eigen::Triplet<double>(points.nv, points.nv, 1.0)); //last entry
    }
    matrix_eigen.resize(system_size, system_size);
    matrix_eigen.setFromTriplets(triplet.begin(), triplet.end());
    matrix_eigen.makeCompressed();
    triplet.clear();
}

void eigen_SOR(Eigen::SparseMatrix<double, Eigen::ColMajor> &matrix, Eigen::VectorXd &X_new, Eigen::VectorXd &X_old, Eigen::VectorXd &source, double omega, int n_iter_max)
{
    Eigen::SparseMatrix<double, Eigen::ColMajor> matrix_low = matrix.triangularView<Eigen::Lower>();
    Eigen::SparseMatrix<double, Eigen::ColMajor> matrix_strict_up = matrix.triangularView<Eigen::StrictlyUpper>();
    Eigen::SparseMatrix<double, Eigen::ColMajor> matrix_rhs = (omega * matrix_strict_up) + ((1.0 - omega) * matrix_low);
    cout << "\n\n";
    for (int iter = 0; iter < n_iter_max; iter++)
    {
        X_new = matrix_low.triangularView<Eigen::Lower>().solve((omega * source) + (matrix_rhs * X_old));
        X_old = X_new;
        double absolute_residual = ((matrix * X_new) - (source / omega)).norm();
        double relative_residual = absolute_residual / source.norm();
        printf("eigen_SOR: iter: %i, n_iter_max: %i, rel_res: %g, abs_res: %g\n", iter, n_iter_max, relative_residual, absolute_residual);
    }
    cout << "\n\n";
}

void loop_SOR(Eigen::SparseMatrix<double, Eigen::ColMajor> &matrix_csc, Eigen::VectorXd &X_new, Eigen::VectorXd &X_old, Eigen::VectorXd &source, double omega, int n_iter_max)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix_csr = convert_csc_to_csr_eigen(matrix_csc);
    write_csv(matrix_csr, "matrix_csr.csv");
    vector<double> diag_terms;
    for (int iv = 0; iv < X_new.size(); iv++)
        diag_terms.push_back(0.0);
    for (int k = 0; k < matrix_csr.outerSize(); ++k)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(matrix_csr, k); it; ++it)
            if (it.row() == it.col())
                diag_terms[it.row()] = it.value();
    for (int iv = 0; iv < X_new.size(); iv++)
        if (fabs(diag_terms[iv]) < 1E-5)
        {
            printf("\n\nloop_SOR diag_terms[%i]: %g\n\n", iv, diag_terms[iv]);
            throw bad_exception();
        }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t t0 = clock();
    double wv = 1.0; //wavenumber

    string meshfile = "/home/shantanu/Desktop/All Simulation Results/Meshless_Methods/CAD_mesh_files/Square/gmsh/Square_n_10_unstruc.msh"; //2D example
    // string meshfile = "/home/shantanu/Desktop/All Simulation Results/Meshless_Methods/CAD_mesh_files/cuboid/Cuboid_n_10_unstruc.msh"; //3D example

    PARAMETERS parameters("parameters_file.csv", meshfile);
    POINTS points(parameters);
    CLOUD cloud(points, parameters);
    Eigen::VectorXd T_num_bicgstab;
    vector<bool> dirichlet_flag;
    for (int iv = 0; iv < points.nv; iv++)
        dirichlet_flag.push_back(false);
    T_num_bicgstab = Eigen::VectorXd::Zero(points.nv + 1); //initialize assuming full Neumann
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv] && dirichlet_flag[iv])
        { //identified at least one boundary point with dirichlet BC (no regularization needed)
            T_num_bicgstab = Eigen::VectorXd::Zero(points.nv);
            break;
        }
    Eigen::VectorXd T_ana = T_num_bicgstab, source = T_num_bicgstab, T_num_eigen_SOR = T_num_bicgstab, T_num_loop_SOR = T_num_bicgstab, T_num_eigen_direct = T_num_bicgstab;
    double max_err, l1_err, grad_x, grad_y, grad_z = 0.0, absolute_residual, relative_residual;
    int dim = parameters.dimension;
    for (int iv = 0; iv < points.nv; iv++)
    { //manufactured solution: sin(wv*x)*sin(wv*y)*sin(wv*z)
        T_ana[iv] = sin(wv * points.xyz[dim * iv]) * sin(wv * points.xyz[dim * iv + 1]);
        if (dim == 3)
            T_ana[iv] = T_ana[iv] * sin(wv * points.xyz[dim * iv + 2]);
    }

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
            source[iv] = wv * wv * dim * T_ana[iv]; //manufactured solution source term
    }

    SOLVER solver_T;
    solver_T.init(points, cloud, parameters, dirichlet_flag, 0.0, 0.0, -1.0, true);
    solver_T.general_solve(points, parameters, T_num_bicgstab, T_num_bicgstab, source);
    if (T_num_bicgstab.rows() == points.nv + 1) //reset level to analytical solution
    {
        T_num_bicgstab = T_num_bicgstab - (Eigen::VectorXd::Ones(points.nv + 1) * (T_num_bicgstab[0] - T_ana[0]));
        T_num_bicgstab[points.nv] = T_ana[points.nv];
    }
    calc_max_l1_relative_error(T_ana, T_num_bicgstab, max_err, l1_err);
    printf("\nEigen BiCGSTAB: relative errors in T: max: %g, avg: %g\n", max_err, l1_err);
    printf("Eigen BiCGSTAB: converged in %i iterations, rel_res: %g, abs_res: %g\n", parameters.n_iter_actual[0], parameters.rel_res_log[0], parameters.abs_res_log[0]);

    Eigen::SparseMatrix<double, Eigen::ColMajor> coeff_matrix;
    assemble_matrix(coeff_matrix, points, cloud, parameters, dirichlet_flag, -1.0, T_ana.rows());

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_eigen_direct;
    solver_eigen_direct.analyzePattern(coeff_matrix);
    solver_eigen_direct.factorize(coeff_matrix);
    T_num_eigen_direct = solver_eigen_direct.solve(source);
    calc_max_l1_relative_error(T_ana, T_num_eigen_direct, max_err, l1_err);
    absolute_residual = (coeff_matrix * T_num_eigen_direct - source).norm();
    relative_residual = absolute_residual / source.norm();
    printf("\nEigen Direct: relative errors in T: max: %g, avg: %g\n", max_err, l1_err);
    printf("Eigen Direct: determinant: %g\n", solver_eigen_direct.determinant());
    printf("Eigen Direct: rel_res: %g, abs_res: %g\n", relative_residual, absolute_residual);

    int n_iter_eigen_SOR = 10;
    eigen_SOR(coeff_matrix, T_num_eigen_SOR, T_num_eigen_SOR, source, 0.4, n_iter_eigen_SOR);
    if (T_num_eigen_SOR.rows() == points.nv + 1) //reset level to analytical solution
    {
        T_num_eigen_SOR = T_num_eigen_SOR - (Eigen::VectorXd::Ones(points.nv + 1) * (T_num_eigen_SOR[0] - T_ana[0]));
        T_num_eigen_SOR[points.nv] = T_ana[points.nv];
    }
    calc_max_l1_relative_error(T_ana, T_num_eigen_SOR, max_err, l1_err);
    absolute_residual = (coeff_matrix * T_num_eigen_SOR - source).norm();
    relative_residual = absolute_residual / source.norm();
    printf("\nEigen SOR: relative errors in T: max: %g, avg: %g\n", max_err, l1_err);
    printf("Eigen SOR: after %i iterations, rel_res: %g, abs_res: %g\n", n_iter_eigen_SOR, relative_residual, absolute_residual);
    // write_csv(points.xyz, points.boundary_flag, T_ana, T_num_eigen_SOR, parameters.dimension, "T_SOR.csv");

    int n_iter_loop_SOR = 10;
    loop_SOR(coeff_matrix, T_num_loop_SOR, T_num_loop_SOR, source, 0.4, n_iter_loop_SOR);
    // if (T_num_loop_SOR.rows() == points.nv + 1) //reset level to analytical solution
    // {
    //     T_num_loop_SOR = T_num_loop_SOR - (Eigen::VectorXd::Ones(points.nv + 1) * (T_num_loop_SOR[0] - T_ana[0]));
    //     T_num_loop_SOR[points.nv] = T_ana[points.nv];
    // }
    // calc_max_l1_relative_error(T_ana, T_num_loop_SOR, max_err, l1_err);
    // double absolute_residual = (coeff_matrix * T_num_loop_SOR - source).norm();
    // double relative_residual = absolute_residual / source.norm();
    // printf("\nLoop SOR: relative errors in T: max: %g, avg: %g\n", max_err, l1_err);
    // printf("Loop SOR: after %i iterations, rel_res: %g, abs_res: %g\n", n_iter_eigen_SOR, relative_residual, absolute_residual);
}