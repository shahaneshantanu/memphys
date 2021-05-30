//Author: Dr. Shantanu Shahane
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "class.hpp"
#include "coefficient_computations.hpp"
#include <unistd.h>
#include <limits.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/MatOp/SparseGenRealShiftSolve.h>
#include "nanoflann.hpp"
using namespace std;

CLOUD::CLOUD(POINTS &points, PARAMETERS &parameters)
{
    clock_t clock_t1 = clock();
    if (parameters.periodic_bc_index.size() == 0)
        calc_cloud_points_fast(points, parameters); //non-periodic case
    else
        calc_cloud_points_fast_periodic_bc(points, parameters);
    parameters.cloud_id_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    clock_t1 = clock();
    re_order_points_reverse_cuthill_mckee(points, parameters);
    parameters.rcm_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    clock_t1 = clock();
    calc_iv_original_nearest_vert(points, parameters);
    calc_charac_dx(points, parameters);
    parameters.cloud_misc_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    calc_grad_laplace_coeffs(points, parameters);
    EIGEN_set_grad_laplace_matrix(points, parameters);
    EIGEN_set_grad_laplace_matrix_separate(points, parameters);
    cout << "\n";
}

void CLOUD::calc_iv_original_nearest_vert(POINTS &points, PARAMETERS &parameters)
{
    for (int iv0 = 0; iv0 < points.nv_original; iv0++)
        points.iv_original_nearest_vert.push_back(-1);
    double x0, y0, z0 = 0.0, x1, y1, z1 = 0.0, dist_square, temp;
    int dim = parameters.dimension, iv_nearest, offset = 0;
    for (int iv0 = 0; iv0 < points.nv_original; iv0++)
    {
        if (points.corner_edge_vertices[iv0])
        { //these points are deleted; thus, nearest vertex has to be found
            x0 = points.xyz_original[dim * iv0], y0 = points.xyz_original[dim * iv0 + 1];
            if (dim == 3)
                z0 = points.xyz_original[dim * iv0 + 2];
            dist_square = INFINITY;
            for (int iv1 = 0; iv1 < points.nv; iv1++)
            {
                // if (points.boundary_flag[iv1])
                // { //[boundary points coupled to boundary]
                x1 = points.xyz[dim * iv1], y1 = points.xyz[dim * iv1 + 1];
                if (dim == 3)
                    z1 = points.xyz[dim * iv1 + 2];
                temp = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) + (z1 - z0) * (z1 - z0);
                if (dist_square >= temp)
                {
                    dist_square = temp;
                    points.iv_original_nearest_vert[iv0] = iv1;
                }
                // }
            }
            offset++; //deleted vertices are offset
        }
        else
        {
            iv_nearest = rcm_points_order[iv0 - offset];
            points.iv_original_nearest_vert[iv0] = iv_nearest;
        }
    }
}

void CLOUD::EIGEN_set_grad_laplace_matrix_separate(POINTS &points, PARAMETERS &parameters)
{
    vector<Eigen::Triplet<double>> triplet;
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
            for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
                triplet.push_back(Eigen::Triplet<double>(iv, nb_points_col[i1], grad_x_coeff[i1]));
    points.grad_x_matrix_EIGEN_boundary.resize(points.nv, points.nv);
    points.grad_x_matrix_EIGEN_boundary.setFromTriplets(triplet.begin(), triplet.end());
    points.grad_x_matrix_EIGEN_boundary.makeCompressed();
    triplet.clear();

    for (int iv = 0; iv < points.nv; iv++)
        if (!points.boundary_flag[iv])
            for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
                triplet.push_back(Eigen::Triplet<double>(iv, nb_points_col[i1], grad_x_coeff[i1]));
    points.grad_x_matrix_EIGEN_internal.resize(points.nv, points.nv);
    points.grad_x_matrix_EIGEN_internal.setFromTriplets(triplet.begin(), triplet.end());
    points.grad_x_matrix_EIGEN_internal.makeCompressed();
    triplet.clear();

    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
            for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
                triplet.push_back(Eigen::Triplet<double>(iv, nb_points_col[i1], grad_y_coeff[i1]));
    points.grad_y_matrix_EIGEN_boundary.resize(points.nv, points.nv);
    points.grad_y_matrix_EIGEN_boundary.setFromTriplets(triplet.begin(), triplet.end());
    points.grad_y_matrix_EIGEN_boundary.makeCompressed();
    triplet.clear();

    for (int iv = 0; iv < points.nv; iv++)
        if (!points.boundary_flag[iv])
            for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
                triplet.push_back(Eigen::Triplet<double>(iv, nb_points_col[i1], grad_y_coeff[i1]));
    points.grad_y_matrix_EIGEN_internal.resize(points.nv, points.nv);
    points.grad_y_matrix_EIGEN_internal.setFromTriplets(triplet.begin(), triplet.end());
    points.grad_y_matrix_EIGEN_internal.makeCompressed();
    triplet.clear();

    if (parameters.dimension == 3)
    {
        for (int iv = 0; iv < points.nv; iv++)
            if (points.boundary_flag[iv])
                for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
                    triplet.push_back(Eigen::Triplet<double>(iv, nb_points_col[i1], grad_z_coeff[i1]));
        points.grad_z_matrix_EIGEN_boundary.resize(points.nv, points.nv);
        points.grad_z_matrix_EIGEN_boundary.setFromTriplets(triplet.begin(), triplet.end());
        points.grad_z_matrix_EIGEN_boundary.makeCompressed();
        triplet.clear();

        for (int iv = 0; iv < points.nv; iv++)
            if (!points.boundary_flag[iv])
                for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
                    triplet.push_back(Eigen::Triplet<double>(iv, nb_points_col[i1], grad_z_coeff[i1]));
        points.grad_z_matrix_EIGEN_internal.resize(points.nv, points.nv);
        points.grad_z_matrix_EIGEN_internal.setFromTriplets(triplet.begin(), triplet.end());
        points.grad_z_matrix_EIGEN_internal.makeCompressed();
        triplet.clear();
    }

    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
            for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
                triplet.push_back(Eigen::Triplet<double>(iv, nb_points_col[i1], laplacian_coeff[i1]));
    points.laplacian_matrix_EIGEN_boundary.resize(points.nv, points.nv);
    points.laplacian_matrix_EIGEN_boundary.setFromTriplets(triplet.begin(), triplet.end());
    points.laplacian_matrix_EIGEN_boundary.makeCompressed();
    triplet.clear();

    for (int iv = 0; iv < points.nv; iv++)
        if (!points.boundary_flag[iv])
            for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
                triplet.push_back(Eigen::Triplet<double>(iv, nb_points_col[i1], laplacian_coeff[i1]));
    points.laplacian_matrix_EIGEN_internal.resize(points.nv, points.nv);
    points.laplacian_matrix_EIGEN_internal.setFromTriplets(triplet.begin(), triplet.end());
    points.laplacian_matrix_EIGEN_internal.makeCompressed();
    triplet.clear();
}

void CLOUD::EIGEN_set_grad_laplace_matrix(POINTS &points, PARAMETERS &parameters)
{
    vector<Eigen::Triplet<double>> triplet;
    for (int iv = 0; iv < points.nv; iv++)
        for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
            triplet.push_back(Eigen::Triplet<double>(iv, nb_points_col[i1], grad_x_coeff[i1]));
    points.grad_x_matrix_EIGEN.resize(points.nv, points.nv);
    points.grad_x_matrix_EIGEN.setFromTriplets(triplet.begin(), triplet.end());
    points.grad_x_matrix_EIGEN.makeCompressed();
    triplet.clear();

    for (int iv = 0; iv < points.nv; iv++)
        for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
            triplet.push_back(Eigen::Triplet<double>(iv, nb_points_col[i1], grad_y_coeff[i1]));
    points.grad_y_matrix_EIGEN.resize(points.nv, points.nv);
    points.grad_y_matrix_EIGEN.setFromTriplets(triplet.begin(), triplet.end());
    points.grad_y_matrix_EIGEN.makeCompressed();
    triplet.clear();

    if (parameters.dimension == 3)
    {
        for (int iv = 0; iv < points.nv; iv++)
            for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
                triplet.push_back(Eigen::Triplet<double>(iv, nb_points_col[i1], grad_z_coeff[i1]));
        points.grad_z_matrix_EIGEN.resize(points.nv, points.nv);
        points.grad_z_matrix_EIGEN.setFromTriplets(triplet.begin(), triplet.end());
        points.grad_z_matrix_EIGEN.makeCompressed();
        triplet.clear();
    }

    for (int iv = 0; iv < points.nv; iv++)
        for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
            triplet.push_back(Eigen::Triplet<double>(iv, nb_points_col[i1], laplacian_coeff[i1]));
    points.laplacian_matrix_EIGEN.resize(points.nv, points.nv);
    points.laplacian_matrix_EIGEN.setFromTriplets(triplet.begin(), triplet.end());
    points.laplacian_matrix_EIGEN.makeCompressed();
    triplet.clear();
}

void CLOUD::calc_grad_laplace_coeffs(POINTS &points, PARAMETERS &parameters)
{
    clock_t t1 = clock(), t2, t3, t4 = clock();
    vector<double> vert;
    vector<int> central_vert_list;
    Eigen::MatrixXd laplacian, grad_x, grad_y, grad_z;
    int dim = parameters.dimension, iv_nb, i1;
    vector<int> ind_p = parameters.periodic_bc_index, iv_sect;
    central_vert_list.push_back(0);
    double scale[3], time, cond_num, xyz_temp[3];
    t3 = clock();
    cout << endl;
    printf("    CLOUD::calc_grad_laplace_coeffs started prints status after every 5 seconds\n");
    for (int iv = 0; iv < points.nv; iv++)
    {
        t2 = clock();
        central_vert_list[0] = 0;                     //coefficient for first vertex needed
        if (parameters.periodic_bc_index.size() == 0) //non-periodic case
            for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
            {
                iv_nb = nb_points_col[i1];
                for (int i = 0; i < dim; i++)
                    vert.push_back(points.xyz[dim * iv_nb + i]);
            }
        else
        {
            iv_sect = points.periodic_bc_section[iv];
            for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
            {
                iv_nb = nb_points_col[i1];
                for (int id = 0; id < dim; id++)
                    xyz_temp[id] = points.xyz[dim * iv_nb + id];
                for (int ip = 0; ip < ind_p.size(); ip++)
                    if (points.periodic_bc_section[iv_nb][ip] == (-iv_sect[ip])) //shift opposite section (nothing happens for iv_sect=0)
                        xyz_temp[ind_p[ip]] = xyz_temp[ind_p[ip]] + (((double)(iv_sect[ip])) * points.xyz_length[ind_p[ip]]);
                for (int id = 0; id < dim; id++)
                    vert.push_back(xyz_temp[id]);
            }
        }
        shifting_scaling(vert, scale, dim);
        cond_num = calc_PHS_RBF_grad_laplace_single_vert(vert, parameters, laplacian, grad_x, grad_y, grad_z, scale, central_vert_list);
        cond_num_RBF.push_back(cond_num);
        vert.clear();

        for (int i1 = 0; i1 < laplacian.size(); i1++)
        { //(nb_points_row[iv + 1] - nb_points_row[iv]) = laplacian.size()
            grad_x_coeff.push_back(grad_x(0, i1));
            grad_y_coeff.push_back(grad_y(0, i1));
            if (dim == 3)
                grad_z_coeff.push_back(grad_z(0, i1));
            laplacian_coeff.push_back(laplacian(0, i1));
        }

        time = ((double)(clock() - t3)) / CLOCKS_PER_SEC;
        if (time > 5.0)
        {
            printf("    CLOUD::calc_grad_laplace_coeffs iv: %i, nv: %i: completed %.2f percent in %g seconds\n", iv, points.nv, 100.0 * iv / points.nv, ((double)(clock() - t1)) / CLOCKS_PER_SEC);
            t3 = clock();
        }
    }
    cout << endl;
    laplacian.resize(0, 0); //free memory
    grad_x.resize(0, 0);    //free memory
    grad_y.resize(0, 0);    //free memory
    grad_z.resize(0, 0);    //free memory

    cond_num_RBF_max = *max_element(cond_num_RBF.begin(), cond_num_RBF.end());
    cond_num_RBF_min = *min_element(cond_num_RBF.begin(), cond_num_RBF.end());
    cond_num_RBF_avg = accumulate(cond_num_RBF.begin(), cond_num_RBF.end(), 0.0) / cond_num_RBF.size();
    printf("CLOUD::calc_grad_laplace_coeffs RBF condition number max: %g, min: %g, avg: %g\n", cond_num_RBF_max, cond_num_RBF_min, cond_num_RBF_avg);
    parameters.grad_laplace_coeff_timer = ((double)(clock() - t4)) / CLOCKS_PER_SEC;
    printf("CLOUD::calc_grad_laplace_coeffs total grad_laplace_coeff time: %g seconds\n", parameters.grad_laplace_coeff_timer);
}

void CLOUD::calc_charac_dx(POINTS &points, PARAMETERS &parameters)
{
    parameters.avg_dx = 0.0;
    parameters.max_dx = 0.0;
    parameters.min_dx = 1E20;
    double local_min_dx, delx, dely, delz = 0.0, dist;
    int iv_nb, isd, dim = parameters.dimension;
    for (int iv = 0; iv < points.nv; iv++)
    {
        local_min_dx = 1E20;
        for (int i1 = nb_points_row[iv]; i1 < nb_points_row[iv + 1]; i1++)
        {
            iv_nb = nb_points_col[i1];
            if (iv != iv_nb)
            {
                delx = points.xyz[dim * iv] - points.xyz[dim * iv_nb];
                dely = points.xyz[dim * iv + 1] - points.xyz[dim * iv_nb + 1];
                if (dim == 3)
                    delz = points.xyz[dim * iv + 2] - points.xyz[dim * iv_nb + 2];
                dist = sqrt(delx * delx + dely * dely + delz * delz);
                if (local_min_dx > dist)
                    local_min_dx = dist;
            }
        }
        parameters.avg_dx += local_min_dx;
        if (parameters.min_dx > local_min_dx)
            parameters.min_dx = local_min_dx;
        if (parameters.max_dx < local_min_dx)
            parameters.max_dx = local_min_dx;
    }
    parameters.avg_dx = parameters.avg_dx / ((double)(points.nv));
    printf("CLOUD::calc_charac_dx Characteristic mesh dx max: %g, min: %g, avg: %g\n", parameters.max_dx, parameters.min_dx, parameters.avg_dx);
}

void CLOUD::re_order_points_reverse_cuthill_mckee(POINTS &points, PARAMETERS &parameters)
{
    vector<int> temp;
    vector<vector<int>> points_adjacency;
    int iv_1, iv_2;
    for (int iv = 0; iv < points.nv; iv++)
        points_adjacency.push_back(temp); //pushback dummy empty vector
    for (iv_1 = 0; iv_1 < points.nv; iv_1++)
    {
        for (int i1 = nb_points_row[iv_1]; i1 < nb_points_row[iv_1 + 1]; i1++)
        {
            iv_2 = nb_points_col[i1];
            points_adjacency[iv_1].push_back(iv_2);
        }
    }
    reverse_cuthill_mckee_ordering(points_adjacency, rcm_points_order);
    re_order_points(points, parameters);

    for (int iv = 0; iv < points.nv; iv++)
        points_adjacency[iv].clear();
    points_adjacency.clear();
}

void CLOUD::re_order_points(POINTS &points, PARAMETERS &parameters)
{
    vector<int> bc_tag_copy;
    vector<double> xyz_copy, normal_copy;
    vector<bool> boundary_flag_copy;
    vector<vector<int>> nb_points_copy, nb_points;

    int new_iv, dim = parameters.dimension;

    xyz_copy = points.xyz;
    for (int iv = 0; iv < points.nv; iv++)
    { //copy xyz co-ordinates
        new_iv = rcm_points_order[iv];
        for (int i = 0; i < dim; i++)
            xyz_copy[dim * new_iv + i] = points.xyz[dim * iv + i];
    }
    points.xyz = xyz_copy; //update xyz co-ordinates
    xyz_copy.clear();

    if (parameters.periodic_bc_index.size() > 0)
    {
        vector<vector<int>> periodic_bc_section_copy;
        periodic_bc_section_copy = points.periodic_bc_section;
        for (int iv = 0; iv < points.nv; iv++)
        { //copy periodic_bc_section
            new_iv = rcm_points_order[iv];
            periodic_bc_section_copy[new_iv] = points.periodic_bc_section[iv];
        }
        points.periodic_bc_section = periodic_bc_section_copy; //update periodic_bc_section
        for (int iv = 0; iv < points.nv; iv++)
            periodic_bc_section_copy[iv].clear();
        periodic_bc_section_copy.clear();

        vector<vector<bool>> periodic_bc_flag_copy;
        periodic_bc_flag_copy = points.periodic_bc_flag;
        for (int iv = 0; iv < points.nv; iv++)
        { //copy periodic_bc_flag
            new_iv = rcm_points_order[iv];
            periodic_bc_flag_copy[new_iv] = points.periodic_bc_flag[iv];
        }
        points.periodic_bc_flag = periodic_bc_flag_copy; //update periodic_bc_flag
        for (int iv = 0; iv < points.nv; iv++)
            periodic_bc_flag_copy[iv].clear();
        periodic_bc_flag_copy.clear();
    }

    normal_copy = points.normal;
    for (int iv = 0; iv < points.nv; iv++)
    { //copy normals
        new_iv = rcm_points_order[iv];
        for (int i = 0; i < dim; i++)
            normal_copy[dim * new_iv + i] = points.normal[dim * iv + i];
    }
    points.normal = normal_copy; //update normal

    bc_tag_copy = points.bc_tag;
    for (int iv = 0; iv < points.nv; iv++)
    { //copy bc_tag
        new_iv = rcm_points_order[iv];
        bc_tag_copy[new_iv] = points.bc_tag[iv];
    }
    points.bc_tag = bc_tag_copy; //update bc_tag
    bc_tag_copy.clear();

    boundary_flag_copy = points.boundary_flag;
    for (int iv = 0; iv < points.nv; iv++)
    { //copy boundary_flag
        new_iv = rcm_points_order[iv];
        boundary_flag_copy[new_iv] = points.boundary_flag[iv];
    }
    points.boundary_flag = boundary_flag_copy; //update boundary_flag
    boundary_flag_copy.clear();

    vector<int> temp;
    for (int iv_1 = 0; iv_1 < points.nv; iv_1++)
    {
        nb_points.push_back(temp);      //initialize with empty vector
        nb_points_copy.push_back(temp); //initialize with empty vector
        for (int i1 = nb_points_row[iv_1]; i1 < nb_points_row[iv_1 + 1]; i1++)
            nb_points[iv_1].push_back(rcm_points_order[nb_points_col[i1]]);
    }
    for (int iv = 0; iv < points.nv; iv++)
    {
        new_iv = rcm_points_order[iv];
        nb_points_copy[new_iv] = nb_points[iv];
    }
    nb_points_row.clear();
    nb_points_col.clear();
    nb_points_row.push_back(0);
    for (int iv0 = 0; iv0 < points.nv; iv0++)
    {
        nb_points_row.push_back(nb_points_row[iv0] + nb_points_copy[iv0].size());
        nb_points_col.insert(nb_points_col.end(), nb_points_copy[iv0].begin(), nb_points_copy[iv0].end());
    }

    for (int iv = 0; iv < points.nv; iv++)
        nb_points[iv].clear();
    nb_points.clear();
    for (int iv = 0; iv < points.nv; iv++)
        nb_points_copy[iv].clear();
    nb_points_copy.clear();
}

void CLOUD::calc_cloud_points_fast_periodic_bc_shifted(POINTS &points, PARAMETERS &parameters, vector<double> &xyz_shifted, vector<int> &periodic_bc_section_value)
{
    PointCloud<double> cloud_nf_for_interior, cloud_nf_for_boundary;
    int dim = parameters.dimension, iv_nb;
    vector<int> ind_p = parameters.periodic_bc_index;
    cloud_nf_for_interior.pts.resize(points.nv), cloud_nf_for_boundary.pts.resize(points.nv);
    for (int iv = 0; iv < points.nv; iv++)
    {
        cloud_nf_for_interior.pts[iv].x = xyz_shifted[dim * iv];
        cloud_nf_for_interior.pts[iv].y = xyz_shifted[dim * iv + 1];
        if (dim == 3)
            cloud_nf_for_interior.pts[iv].z = xyz_shifted[dim * iv + 2];
        else
            cloud_nf_for_interior.pts[iv].z = 0.0; //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
    }
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (!points.boundary_flag[iv])
        { //internal points are coupled with boundary points
            cloud_nf_for_boundary.pts[iv].x = xyz_shifted[dim * iv];
            cloud_nf_for_boundary.pts[iv].y = xyz_shifted[dim * iv + 1];
            if (dim == 3)
                cloud_nf_for_boundary.pts[iv].z = xyz_shifted[dim * iv + 2];
            else
                cloud_nf_for_boundary.pts[iv].z = 0.0; //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
        }
        else
        { //all boundary co-ordinates set to infinity so that they are never coupled with any boundary point
            cloud_nf_for_boundary.pts[iv].x = numeric_limits<double>::infinity();
            cloud_nf_for_boundary.pts[iv].y = numeric_limits<double>::infinity();
            cloud_nf_for_boundary.pts[iv].z = numeric_limits<double>::infinity(); //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
        }
    }
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3> nanoflann_kd_tree_for_interior;
    nanoflann_kd_tree_for_interior index_for_interior(3, cloud_nf_for_interior, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index_for_interior.buildIndex();

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3> nanoflann_kd_tree_for_boundary;
    nanoflann_kd_tree_for_boundary index_for_boundary(3, cloud_nf_for_boundary, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index_for_boundary.buildIndex();

    vector<size_t> nb_vert(parameters.cloud_size);
    vector<double> nb_dist(parameters.cloud_size);
    double query_pt[3];
    for (int iv = 0; iv < points.nv; iv++)
        if (points.periodic_bc_section[iv] == periodic_bc_section_value)
        {
            query_pt[0] = xyz_shifted[dim * iv], query_pt[1] = xyz_shifted[dim * iv + 1];
            if (dim == 3)
                query_pt[2] = xyz_shifted[dim * iv + 2];
            else
                query_pt[2] = 0.0; //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
            if (points.boundary_flag[iv])
            {
                index_for_boundary.knnSearch(&query_pt[0], parameters.cloud_size, &nb_vert[0], &nb_dist[0]);
                nb_points_col[nb_points_row[iv]] = iv;
                for (int i1 = 0; i1 < nb_vert.size() - 1; i1++)
                { //first entry is "iv": hence "nb_vert.size() - 1"
                    iv_nb = nb_vert[i1];
                    if (points.boundary_flag[iv_nb])
                    {
                        cout << "\n\nERROR from CLOUD::calc_cloud_points_fast boundary iv: " << iv << " (boundary_flag[iv]: " << points.boundary_flag[iv] << ") tried to couple to a boundary vertex: " << iv_nb << " (boundary_flag[iv_nb]: " << points.boundary_flag[iv_nb] << ") \n\n";
                        throw bad_exception();
                    }
                    else
                        nb_points_col[nb_points_row[iv] + i1 + 1] = iv_nb; //first entry is "iv"
                }
            }
            else
            { //internal points
                index_for_interior.knnSearch(&query_pt[0], parameters.cloud_size, &nb_vert[0], &nb_dist[0]);
                for (int i1 = 0; i1 < nb_vert.size(); i1++)
                    nb_points_col[nb_points_row[iv] + i1] = nb_vert[i1];
            }
        }
    cloud_nf_for_interior.pts.resize(0), cloud_nf_for_boundary.pts.resize(0);
}

void CLOUD::calc_cloud_points_fast_periodic_bc(POINTS &points, PARAMETERS &parameters)
{ //Uses KD-Tree algorithm from Nanoflann (https://github.com/jlblancoc/nanoflann)
    nb_points_row.push_back(0);
    for (int iv = 0; iv < points.nv; iv++)
        nb_points_row.push_back(nb_points_row[iv] + parameters.cloud_size);
    for (int i1 = 0; i1 < points.nv * parameters.cloud_size; i1++)
        nb_points_col.push_back(-1);
    int dim = parameters.dimension;
    vector<int> ind_p = parameters.periodic_bc_index, empty_int;

    vector<vector<int>> section_list;
    for (int i1 = 0; i1 < ((int)(pow(3, ind_p.size()))); i1++)
        section_list.push_back(empty_int);
    if (ind_p.size() == 1) //section_list = [[-1], [0], [1]]
        section_list[0].push_back(-1), section_list[1].push_back(0), section_list[2].push_back(1);
    else if (ind_p.size() == 2) //section_list = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
        for (int i1 = 0; i1 < 3; i1++)
            for (int i2 = 0; i2 < 3; i2++)
                section_list[3 * i1 + i2].push_back(i1 - 1), section_list[3 * i1 + i2].push_back(i2 - 1);
    else if (ind_p.size() == 3 && dim == 3)
        for (int i1 = 0; i1 < 3; i1++)
            for (int i2 = 0; i2 < 3; i2++)
                for (int i3 = 0; i3 < 3; i3++)
                {
                    section_list[9 * i1 + 3 * i2 + i3].push_back(i1 - 1);
                    section_list[9 * i1 + 3 * i2 + i3].push_back(i2 - 1);
                    section_list[9 * i1 + 3 * i2 + i3].push_back(i3 - 1);
                }
    else
    {
        cout << "\n\nCLOUD::calc_cloud_points_fast_periodic_bc number of periodic axes ind_p.size(): " << ind_p.size() << " should not be greater than problem dimension: " << dim << "\n\n";
        throw bad_exception();
    }

    vector<double> xyz_shifted;
    int i_sec;
    for (int i1 = 0; i1 < section_list.size(); i1++)
    {
        xyz_shifted = points.xyz;
        for (int ip = 0; ip < ind_p.size(); ip++)
        {
            i_sec = section_list[i1][ip];
            for (int iv = 0; iv < points.nv; iv++)
                if (points.periodic_bc_section[iv][ip] == -i_sec) //shift opposite section (nothing happens for i_sec=0)
                    xyz_shifted[dim * iv + ind_p[ip]] = xyz_shifted[dim * iv + ind_p[ip]] + (((double)(i_sec)) * points.xyz_length[ind_p[ip]]);
            calc_cloud_points_fast_periodic_bc_shifted(points, parameters, xyz_shifted, section_list[i1]);
        }
    }
    xyz_shifted.clear();
}

void CLOUD::calc_cloud_points_fast(POINTS &points, PARAMETERS &parameters)
{ //Uses KD-Tree algorithm from Nanoflann (https://github.com/jlblancoc/nanoflann)
    PointCloud<double> cloud_nf_for_interior, cloud_nf_for_boundary;
    int dim = parameters.dimension;
    cloud_nf_for_interior.pts.resize(points.nv);
    cloud_nf_for_boundary.pts.resize(points.nv);
    for (int iv = 0; iv < points.nv; iv++)
    {
        cloud_nf_for_interior.pts[iv].x = points.xyz[dim * iv];
        cloud_nf_for_interior.pts[iv].y = points.xyz[dim * iv + 1];
        if (dim == 3)
            cloud_nf_for_interior.pts[iv].z = points.xyz[dim * iv + 2];
        else
            cloud_nf_for_interior.pts[iv].z = 0.0; //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
    }
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (!points.boundary_flag[iv])
        { //internal points are coupled with boundary points
            cloud_nf_for_boundary.pts[iv].x = points.xyz[dim * iv];
            cloud_nf_for_boundary.pts[iv].y = points.xyz[dim * iv + 1];
            if (dim == 3)
                cloud_nf_for_boundary.pts[iv].z = points.xyz[dim * iv + 2];
            else
                cloud_nf_for_boundary.pts[iv].z = 0.0; //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
        }
        else
        { //all boundary co-ordinates set to infinity so that they are never coupled with any boundary point
            cloud_nf_for_boundary.pts[iv].x = numeric_limits<double>::infinity();
            cloud_nf_for_boundary.pts[iv].y = numeric_limits<double>::infinity();
            cloud_nf_for_boundary.pts[iv].z = numeric_limits<double>::infinity(); //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
        }
    }

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3> nanoflann_kd_tree_for_interior;
    nanoflann_kd_tree_for_interior index_for_interior(3, cloud_nf_for_interior, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index_for_interior.buildIndex();

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3> nanoflann_kd_tree_for_boundary;
    nanoflann_kd_tree_for_boundary index_for_boundary(3, cloud_nf_for_boundary, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index_for_boundary.buildIndex();

    vector<size_t> nb_vert(parameters.cloud_size);
    vector<double> nb_dist(parameters.cloud_size);
    double query_pt[3];
    int iv_nb;
    nb_points_row.push_back(0);
    for (int iv = 0; iv < points.nv; iv++)
    {
        query_pt[0] = points.xyz[dim * iv], query_pt[1] = points.xyz[dim * iv + 1];
        if (dim == 3)
            query_pt[2] = points.xyz[dim * iv + 2];
        else
            query_pt[2] = 0.0; //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
        if (points.boundary_flag[iv])
        {
            index_for_boundary.knnSearch(&query_pt[0], parameters.cloud_size, &nb_vert[0], &nb_dist[0]);
            nb_points_row.push_back(nb_points_row[iv] + parameters.cloud_size);
            nb_points_col.push_back(iv);
            for (int i1 = 0; i1 < nb_vert.size() - 1; i1++)
            { //first entry is "iv": hence "nb_vert.size() - 1"
                iv_nb = nb_vert[i1];
                if (points.boundary_flag[iv_nb])
                {
                    cout << "\n\nERROR from CLOUD::calc_cloud_points_fast boundary iv: " << iv << " (boundary_flag[iv]: " << points.boundary_flag[iv] << ") tried to couple to a boundary vertex: " << iv_nb << " (boundary_flag[iv_nb]: " << points.boundary_flag[iv_nb] << ") \n\n";
                    throw bad_exception();
                }
                else
                    nb_points_col.push_back(iv_nb);
            }
        }
        else
        { //internal points
            index_for_interior.knnSearch(&query_pt[0], parameters.cloud_size, &nb_vert[0], &nb_dist[0]);
            nb_points_row.push_back(nb_points_row[iv] + parameters.cloud_size);
            nb_points_col.insert(nb_points_col.end(), nb_vert.begin(), nb_vert.end());
        }
    }
}

void CLOUD::calc_cloud_points_slow(POINTS &points, PARAMETERS &parameters)
{ //calculate neighboring points for all vertices: Computations: Order(points.nv^2)
    vector<double> dist_square, k_min_dist_square;
    vector<int> k_min_points;
    for (int iv = 0; iv < points.nv; iv++)
        dist_square.push_back(0.0); //initialize
    double x0, y0, z0;
    int k = parameters.cloud_size, dim = parameters.dimension;
    nb_points_row.push_back(0);
    if (dim == 2)
    { //2D problem
        for (int iv0 = 0; iv0 < points.nv; iv0++)
        {
            x0 = points.xyz[dim * iv0];
            y0 = points.xyz[dim * iv0 + 1];
            for (int iv = 0; iv < points.nv; iv++)
            {
                if (!points.boundary_flag[iv0])
                { //all points added if iv0 is internal
                    dist_square[iv] = (x0 - points.xyz[dim * iv]) * (x0 - points.xyz[dim * iv]);
                    dist_square[iv] = dist_square[iv] + (y0 - points.xyz[dim * iv + 1]) * (y0 - points.xyz[dim * iv + 1]);
                }
                else
                { //iv0 is boundary: only couple with internal points
                    if (iv0 == iv || !points.boundary_flag[iv])
                    { //self-coupling OR only internal points
                        dist_square[iv] = (x0 - points.xyz[dim * iv]) * (x0 - points.xyz[dim * iv]);
                        dist_square[iv] = dist_square[iv] + (y0 - points.xyz[dim * iv + 1]) * (y0 - points.xyz[dim * iv + 1]);
                    }
                    else
                        dist_square[iv] = numeric_limits<double>::infinity(); //iv is boundary and not equal to iv0 (thus should not be coupled)
                }
            }
            k_smallest_elements(k_min_dist_square, k_min_points, dist_square, k);
            nb_points_row.push_back(nb_points_row[iv0] + k_min_points.size());
            nb_points_col.insert(nb_points_col.end(), k_min_points.begin(), k_min_points.end());
        }
    }
    else
    { //3D problem
        for (int iv0 = 0; iv0 < points.nv; iv0++)
        {
            x0 = points.xyz[dim * iv0];
            y0 = points.xyz[dim * iv0 + 1];
            z0 = points.xyz[dim * iv0 + 2];
            for (int iv = 0; iv < points.nv; iv++)
            {
                if (!points.boundary_flag[iv0])
                { //all points added if iv0 is internal
                    dist_square[iv] = (x0 - points.xyz[dim * iv]) * (x0 - points.xyz[dim * iv]);
                    dist_square[iv] = dist_square[iv] + (y0 - points.xyz[dim * iv + 1]) * (y0 - points.xyz[dim * iv + 1]);
                    dist_square[iv] = dist_square[iv] + (z0 - points.xyz[dim * iv + 2]) * (z0 - points.xyz[dim * iv + 2]);
                }
                else
                { //iv0 is boundary: only couple with internal points
                    if (iv0 == iv || !points.boundary_flag[iv])
                    { //self-coupling OR only internal points
                        dist_square[iv] = (x0 - points.xyz[dim * iv]) * (x0 - points.xyz[dim * iv]);
                        dist_square[iv] = dist_square[iv] + (y0 - points.xyz[dim * iv + 1]) * (y0 - points.xyz[dim * iv + 1]);
                        dist_square[iv] = dist_square[iv] + (z0 - points.xyz[dim * iv + 2]) * (z0 - points.xyz[dim * iv + 2]);
                    }
                    else
                        dist_square[iv] = numeric_limits<double>::infinity(); //iv is boundary and not equal to iv0 (thus should not be coupled)
                }
            }
            k_smallest_elements(k_min_dist_square, k_min_points, dist_square, k);
            nb_points_row.push_back(nb_points_row[iv0] + k_min_points.size());
            nb_points_col.insert(nb_points_col.end(), k_min_points.begin(), k_min_points.end());
        }
    }
    dist_square.clear();
    k_min_dist_square.clear();
    k_min_points.clear();
}