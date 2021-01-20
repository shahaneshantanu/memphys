//Author: Dr. Shantanu Shahane
#ifndef class_H_ /* Include guard */
#define class_H_
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>
#include <Eigen/Core>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "general_functions.hpp"
using namespace std;

const int INTERIOR_TAG = -1;
// const int INLET_BC_TAG = 1;
// const int OUTLET_BC_TAG = 2;
// const int WALL_BC_TAG = 3;
// const int SYMMETRY_BC_TAG = 4;

template <typename T>
struct PointCloud
{ //taken from Nanoflann library: "utils.h": https://github.com/jlblancoc/nanoflann/blob/master/examples/utils.h
    struct Point
    {
        T x, y, z;
    };
    std::vector<Point> pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); } // Must return the number of data points
    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }
    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
};

class PARAMETERS
{
public:
    int cloud_size;                            //cloud size
    double cloud_size_multiplier;              //cloud size: cloud_size_multiplier * num_poly_terms
    int phs_deg;                               //degree of the polyharmonic spline: r^(phs_deg)
    int poly_deg;                              //degree of the polynomial to be appended
    int num_poly_terms;                        //number of the polynomial terms to be appended
    int dimension;                             //problem dimension (can be either 2 or 3 only)
    Eigen::MatrixXi polynomial_term_exponents; //exponents of the polynomial terms
    string meshfile;                           //meshfile name
    string output_file_prefix;                 //used to write all output files
    double max_dx, min_dx, avg_dx;             //characteristic length scales of mesh
    double dt;
    double rho = -10.0, mu = rho; //should be set in the main file
    double steady_tolerance, solver_tolerance, Courant, precond_droptol;
    int nt, euclid_precond_level_hypre, gmres_kdim, n_iter;
    string solver_type; //hypre_ilu_gmres, eigen_direct, eigen_ilu_bicgstab

    double grad_x_eigval_real, grad_x_eigval_imag, grad_y_eigval_real, grad_y_eigval_imag, grad_z_eigval_real, grad_z_eigval_imag, laplace_eigval_real, laplace_eigval_imag;
    double cloud_id_timer, rcm_timer, cloud_misc_timer, points_timer, grad_laplace_coeff_timer, factoring_timer, solve_timer, total_timer;
    int nt_actual = 0;

    vector<double> rel_res_log, abs_res_log, regul_alpha_log, steady_error_log; //logs of size nt_actual
    vector<int> n_iter_actual;                                                  //logs of size nt_actual

public:
    PARAMETERS(string parameter_file, string gmsh_file);
    void read_calc_parameters(string parameter_file);
    void verify_parameters();
    void calc_cloud_num_points();
    void get_problem_dimension_msh();
    void calc_polynomial_term_exponents();
    void calc_dt(Eigen::SparseMatrix<double, Eigen::RowMajor> &grad_x, Eigen::SparseMatrix<double, Eigen::RowMajor> &grad_y, Eigen::SparseMatrix<double, Eigen::RowMajor> &grad_z, Eigen::SparseMatrix<double, Eigen::RowMajor> &laplacian, double u0, double v0, double w0, double alpha);
};

class POINTS
{ //uses Compressed Row Format of Sparse Matrices
public:
    vector<bool> corner_edge_vertices;                                   //true at corner (for 2D, 3D problems) and edge (for 3D problems only): size [nv]
    int nv;                                                              //Number of vertices in original msh file
    int nv_original;                                                     //Number of vertices
    vector<double> xyz;                                                  //vertex co-ordinates: size [nv X dim]
    vector<double> xyz_original;                                         //all vertex co-ordinates in original msh file: size [nv_original X dim]; nv_original can be greater than nv
    int nelem_original;                                                  //no. of elements in the original file
    vector<bool> elem_boundary_flag_original;                            //boundary_flag=1 if elem is on boundary; else boundary_flag=0: size [nelem_original]
    vector<vector<int>> elem_vert_original;                              //connectivity in original msh file
    vector<double> boundary_face_area_original;                          //area or length of boundary elements for 3D or 2D (used to compute fluxes at boundaries): size [nelem_original]
    vector<int> elem_bc_tag_original;                                    //bc_tag associated with each element (helps to identify various boundary areas and internal region): size [nelem_original]
    vector<int> iv_original_nearest_vert;                                //nearest vertex no for vertices in xyz_original: size [nv_original]; nv_original can be greater than nv
    vector<double> normal;                                               //vertex co-ordinates (relavant only for boundary vertices): size [nv X dim]
    vector<bool> boundary_flag;                                          //boundary_flag=1 if it is on boundary; else boundary_flag=0: size [nv]
    vector<int> bc_tag;                                                  //bc_tag associated with each vertex (helps to identify various boundary areas and internal region): size [nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> grad_x_matrix_EIGEN;    //used for convection source term size [points.nv X points.nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> grad_y_matrix_EIGEN;    //used for convection source term size [points.nv X points.nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> grad_z_matrix_EIGEN;    //used for convection source term size [points.nv X points.nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> laplacian_matrix_EIGEN; //used for diffusion source term size [points.nv X points.nv]

    Eigen::SparseMatrix<double, Eigen::RowMajor> grad_x_matrix_EIGEN_boundary, grad_x_matrix_EIGEN_internal;       //[points.nv X points.nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> grad_y_matrix_EIGEN_boundary, grad_y_matrix_EIGEN_internal;       //[points.nv X points.nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> grad_z_matrix_EIGEN_boundary, grad_z_matrix_EIGEN_internal;       //[points.nv X points.nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> laplacian_matrix_EIGEN_boundary, laplacian_matrix_EIGEN_internal; //[points.nv X points.nv]

public:
    POINTS(PARAMETERS &parameters);
    void read_points_xyz_msh(PARAMETERS &parameters);
    void read_points_flag_msh(PARAMETERS &parameters);
    void calc_vert_normal(PARAMETERS &parameters);
    void calc_boundary_face_area(PARAMETERS &parameters);
    void calc_elem_bc_tag(PARAMETERS &parameters);
    void read_elem_vert_complete_msh(PARAMETERS &parameters, vector<vector<int>> &elem_vert, vector<bool> &elem_boundary_flag);
    void calc_vert_nb_cv(PARAMETERS &parameters, vector<vector<int>> &vert_nb_cv, vector<vector<int>> &elem_vert);
    void calc_elem_normal_2D(vector<double> &elem_normal, vector<vector<int>> &vert_nb_cv, vector<vector<int>> &elem_vert, vector<bool> &elem_boundary_flag);
    void calc_elem_normal_3D(vector<double> &elem_normal, vector<vector<int>> &vert_nb_cv, vector<vector<int>> &elem_vert, vector<bool> &elem_boundary_flag);
    void delete_corner_edge_vertices(PARAMETERS &parameters);
};

class CLOUD
{
public:
    vector<int> nb_points_row;      //neighboring points of each point: size: nv+1
    vector<int> nb_points_col;      //neighboring points of each point: size: nb_points_row[nv]
    vector<double> grad_x_coeff;    //coefficient for grad_x at each point: base: (CLOUD.nb_points_row, CLOUD.nb_points_col)
    vector<double> grad_y_coeff;    //coefficient for grad_y at each point: base: (CLOUD.nb_points_row, CLOUD.nb_points_col)
    vector<double> grad_z_coeff;    //coefficient for grad_z at each point: base: (CLOUD.nb_points_row, CLOUD.nb_points_col)
    vector<double> laplacian_coeff; //coefficient for laplacian at each point: base: (CLOUD.nb_points_row, CLOUD.nb_points_col)
    vector<double> cond_num_RBF;    //condition number of RBF A matrix for each point: size [nv]
    double cond_num_RBF_max;        //statistics of cond_num_RBF
    double cond_num_RBF_min;        //statistics of cond_num_RBF
    double cond_num_RBF_avg;        //statistics of cond_num_RBF
    vector<int> rcm_points_order;   //reordering list obtained by RCM algorithm
public:
    CLOUD(POINTS &points, PARAMETERS &parameters);
    void calc_cloud_points_slow(POINTS &points, PARAMETERS &parameters);
    void calc_cloud_points_fast(POINTS &points, PARAMETERS &parameters);
    void re_order_points_reverse_cuthill_mckee(POINTS &points, PARAMETERS &parameters);
    void re_order_points(POINTS &points, PARAMETERS &parameters);
    void calc_iv_original_nearest_vert(POINTS &points, PARAMETERS &parameters);
    void calc_charac_dx(POINTS &points, PARAMETERS &parameters);
    void calc_grad_laplace_coeffs(POINTS &points, PARAMETERS &parameters);
    void EIGEN_set_grad_laplace_matrix(POINTS &points, PARAMETERS &parameters);
    void EIGEN_set_grad_laplace_matrix_separate(POINTS &points, PARAMETERS &parameters);
};

class SOLVER
{
public:
    // Functions
    void init(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &dirichlet_flag, double unsteady_term_coeff, double conv_term_coeff, double diff_term_coeff, bool log_flag);
    void HYPRE_set_coeff(PARAMETERS &parameters);
    void EIGEN_set_coeff();
    void general_solve(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &field_new, Eigen::VectorXd &field_old, Eigen::VectorXd &rhs);
    void set_solve_parameters();
    void calc_coeff_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);
    void scale_coeff(POINTS &points);

    // Variables
    HYPRE_IJMatrix coeff_HYPRE;
    HYPRE_ParCSRMatrix parcsr_coeff_HYPRE;
    HYPRE_IJVector source_HYPRE;
    HYPRE_ParVector par_source_HYPRE;
    HYPRE_IJVector X_HYPRE;
    HYPRE_ParVector par_X_HYPRE;
    HYPRE_Solver solver_X_HYPRE, precond_X_HYPRE;
    Eigen::SparseMatrix<double, Eigen::RowMajor> coeff_EIGEN;
    Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::COLAMDOrdering<int>> solver_eigen_direct;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::IncompleteLUT<double>> solver_eigen_ilu_bicgstab;

    vector<tuple<int, int, double>> coeff_matrix; //store coefficient matrix with sparsity structure of (row, col, value)
    vector<double> scale;

    int *rows_HYPRE; //array of size ncv going from 0 to ncv-1
    double *X;       //unknown vector
    double *source;  //source term

    string solver_type; //hypre_ilu_gmres, eigen_direct, eigen_ilu_bicgstab
    double unsteady_term_coeff_1, conv_term_coeff_1, diff_term_coeff_1;
    int print_flag = 0;             //decide how much to print during solution
    int n_iter;                     //max solver iterations
    int euclid_precond_level_hypre; //level setting in Euclid pre-conditioner
    double precond_droptol;         //setting in Euclid pre-conditioner
    int gmres_kdim;                 //GMRES max size of Krylov subspace
    double solver_tolerance;        //solver tolerance
    double l2_norm;
    vector<bool> dirichlet_flag_1; //true: use Dirichlet BC, else use Neumann BC
    int system_size;               //no. of unknowns
    bool log_flag_1 = true;
};

class FRACTIONAL_STEP_1
{ //hat velocity formulation
public:
    // Functions
    FRACTIONAL_STEP_1(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &u_dirichlet_flag1, vector<bool> &v_dirichlet_flag1, vector<bool> &p_dirichlet_flag1, int temporal_order1);
    void check_bc(POINTS &points, PARAMETERS &parameters);
    double single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, int it1);
    double single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y, int it1);
    void calc_vel_hat(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void calc_vel_hat(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y);
    void calc_pressure(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void calc_vel_corr(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old);
    double extras(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);

    // Variables
    SOLVER solver_p;
    Eigen::VectorXd zero_vector, zero_vector_1;
    Eigen::VectorXd uh, vh;
    Eigen::VectorXd p_source;
    Eigen::VectorXd u_source_old, v_source_old;
    Eigen::VectorXd u_source_old_old, v_source_old_old; //used only for multi-step method
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    bool p_bc_full_neumann;
    int temporal_order = -1, it;
};

class SOLIDIFICATION
{
public:
    // Functions
    SOLIDIFICATION(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, int temporal_order1);
    void single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &T_new, Eigen::VectorXd &T_old, Eigen::VectorXd &fs_new, Eigen::VectorXd &fs_old, int it);
    void write_tecplot_temporal_header(POINTS &points, PARAMETERS &parameters);
    void write_tecplot_temporal_fields(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &T_new, Eigen::VectorXd &fs_new, int it);

    // Variables
    Eigen::VectorXd dfs_dT, T_source;
    Eigen::VectorXd T_old_old, fs_old_old; //used only for multi-step method
    double Tliq = 915.0;
    double Tsol = 850.0;
    double Tf = 935.2;
    double Teps = 2.0;
    double k_partition = 0.13;
    double Lf = 390000.0;
    double Cp = 1006.0;
    double alpha = 5E-5;
    int temporal_order = -1;
};

#endif