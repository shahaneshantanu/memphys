//Author: Dr. Shantanu Shahane
#ifndef postprocessing_functions_H_ /* Include guard */
#define postprocessing_functions_H_
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
#include <unistd.h>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "general_functions.hpp"
#include "class.hpp"
using namespace std;

double calc_boundary_flux(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &phi_x, Eigen::VectorXd &phi_y, int bc_tag);

double calc_boundary_flux(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &phi_x, Eigen::VectorXd &phi_y, Eigen::VectorXd &phi_z, int bc_tag);

void write_simulation_details(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);

void write_iteration_details(PARAMETERS &parameters);

void write_navier_stokes_errors_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_ana, Eigen::VectorXd &v_ana, Eigen::VectorXd &p_ana, Eigen::VectorXd &u_num, Eigen::VectorXd &v_num, Eigen::VectorXd &p_num);

void write_navier_stokes_residuals_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p, string output_file_suffix);

void write_navier_stokes_tecplot_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_ana, Eigen::VectorXd &v_ana, Eigen::VectorXd &p_ana, Eigen::VectorXd &u_num, Eigen::VectorXd &v_num, Eigen::VectorXd &p_num);

void write_temperature_tecplot_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &T_ana, Eigen::VectorXd &T_num);

void write_navier_stokes_tecplot_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_num, Eigen::VectorXd &v_num, Eigen::VectorXd &p_num);

void write_navier_stokes_tecplot_temporal_header(POINTS &points, PARAMETERS &parameters);

void write_navier_stokes_tecplot_temporal_fields(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_num, Eigen::VectorXd &v_num, Eigen::VectorXd &w_num, Eigen::VectorXd &p_num, int it);

void write_temperature_tecplot_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &T_num);

void write_temperature_tecplot_3D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &T_ana, Eigen::VectorXd &T_num);

void write_temperature_tecplot_3D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &T_num);

#endif