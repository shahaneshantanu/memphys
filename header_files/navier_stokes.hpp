//Author: Dr. Shantanu Shahane
#ifndef navier_stokes_H_ /* Include guard */
#define navier_stokes_H_
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
#include "class.hpp"
#include "postprocessing_functions.hpp"
using namespace std;

void calc_navier_stokes_residuals_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p);

void calc_navier_stokes_residuals_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y);

void calc_navier_stokes_errors_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_ana, Eigen::VectorXd &v_ana, Eigen::VectorXd &p_ana, Eigen::VectorXd &u_num, Eigen::VectorXd &v_num, Eigen::VectorXd &p_num);

void calc_navier_stokes_residuals_3D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &w, Eigen::VectorXd &p);

void calc_navier_stokes_residuals_3D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &w, Eigen::VectorXd &p, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y, Eigen::VectorXd &body_force_z);

void calc_navier_stokes_errors_3D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_ana, Eigen::VectorXd &v_ana, Eigen::VectorXd &w_ana, Eigen::VectorXd &p_ana, Eigen::VectorXd &u_num, Eigen::VectorXd &v_num, Eigen::VectorXd &w_num, Eigen::VectorXd &p_num);

#endif