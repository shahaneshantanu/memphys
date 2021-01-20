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
using namespace std;

POINTS::POINTS(PARAMETERS &parameters)
{
    cout << "\n";
    clock_t clock_t1 = clock();
    read_points_xyz_msh(parameters);
    read_points_flag_msh(parameters);
    calc_vert_normal(parameters);
    calc_boundary_face_area(parameters);
    delete_corner_edge_vertices(parameters);
    cout << "POINTS::read_points_xyz_msh original mesh nv: " << nv_original << ", nelem: " << nelem_original << endl;
    cout << "POINTS::POINTS after deleting corners nv: " << nv << endl;
    cout << "\n";
    for (int iv = 0; iv < nv; iv++)
        bc_tag.push_back(INTERIOR_TAG); //initialize to interior; should be set to correct value in main file if required
    for (int icv = 0; icv < nelem_original; icv++)
        elem_bc_tag_original.push_back(INTERIOR_TAG); //initialize to interior; should be set to correct value by calling points.calc_elem_bc_tag(parameters) in main file if required
    parameters.points_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
}

void POINTS::calc_elem_bc_tag(PARAMETERS &parameters)
{
    int iv_orig, iv_new, dim = parameters.dimension;
    double dist = 0.0;
    vector<int> iv_tag_temp;
    vector<int>::iterator it;
    for (int icv = 0; icv < nelem_original; icv++)
    {
        if (elem_boundary_flag_original[icv])
        {
            for (int i1 = 0; i1 < elem_vert_original[icv].size(); i1++)
            {
                iv_orig = elem_vert_original[icv][i1];
                iv_new = iv_original_nearest_vert[iv_orig];
                dist = 0.0;
                for (int i2 = 0; i2 < dim; i2++)
                    dist = dist + ((xyz[dim * iv_new + i2] - xyz_original[dim * iv_orig + i2]) * (xyz[dim * iv_new + i2] - xyz_original[dim * iv_orig + i2]));
                if (sqrt(dist) < 1E-5) //keep vertices which are not removed (thus, dist should be zero)
                    iv_tag_temp.push_back(bc_tag[iv_new]);
            }
            if (iv_tag_temp.size() == 0)
            {
                printf("\n\nERROR from POINTS::calc_elem_bc_tag icv: %i does not have any vertices which are not deleted; elem_vert_original[icv].size(): %lu\n\n", icv, elem_vert_original[icv].size());
                throw bad_exception();
            }
            sort(iv_tag_temp.begin(), iv_tag_temp.end());          //sort vector
            it = unique(iv_tag_temp.begin(), iv_tag_temp.end());   //get indices of duplicate entries
            iv_tag_temp.erase(it, iv_tag_temp.end());              //delete duplicate entries
            iv_tag_temp.resize(distance(iv_tag_temp.begin(), it)); //resize vector to remove empty entry locations
            if (iv_tag_temp.size() > 1)
            {
                printf("\n\nERROR from POINTS::calc_elem_bc_tag icv: %i has vertices belonging to %lu bc_tag; elem_vert_original[icv].size(): %lu\n\n", icv, iv_tag_temp.size(), elem_vert_original[icv].size());
                throw bad_exception();
            }
            elem_bc_tag_original[icv] = iv_tag_temp[0];

            iv_tag_temp.clear();
        }
    }
}

void POINTS::calc_boundary_face_area(PARAMETERS &parameters)
{ //area or length of boundary elements for 3D or 2D (used to compute fluxes at boundaries)
    //algorithm references: http://geomalgorithms.com/a01-_area.html#3D%20Polygons, https://stackoverflow.com/questions/12642256/python-find-area-of-polygon-from-xyz-coordinates
    double crossprod_sum[3], crossprod[3], v1[3], v2[3], normal[3];
    int iv1, iv2, dim = parameters.dimension;
    vector<vector<int>> vert_nb_cv;
    vector<double> elem_normal;
    if (dim == 3)
    {
        calc_vert_nb_cv(parameters, vert_nb_cv, elem_vert_original);
        calc_elem_normal_3D(elem_normal, vert_nb_cv, elem_vert_original, elem_boundary_flag_original);
    }
    for (int icv = 0; icv < elem_vert_original.size(); icv++)
    {
        boundary_face_area_original.push_back(0.0);
        if (elem_boundary_flag_original[icv])
        {
            if (dim == 3)
            { //3D: area of face
                crossprod_sum[0] = 0.0, crossprod_sum[1] = 0.0, crossprod_sum[2] = 0.0;
                for (int i = 0; i < elem_vert_original[icv].size(); i++)
                {
                    iv1 = elem_vert_original[icv][i];
                    if (i < elem_vert_original[icv].size() - 1)
                        iv2 = elem_vert_original[icv][i + 1];
                    else
                        iv2 = elem_vert_original[icv][0]; //last vertex gets connected to vertex no "0"
                    for (int j = 0; j < 3; j++)
                    {
                        v1[j] = xyz_original[3 * iv1 + j];
                        v2[j] = xyz_original[3 * iv2 + j];
                    }
                    cross_product(crossprod, v1, v2);
                    for (int j = 0; j < 3; j++)
                        crossprod_sum[j] += crossprod[j];
                }
                for (int i = 0; i < 3; i++)
                    normal[i] = elem_normal[3 * icv + i];
                boundary_face_area_original[icv] = 0.5 * fabs(crossprod_sum[0] * normal[0] + crossprod_sum[1] * normal[1] + crossprod_sum[2] * normal[2]);
            }
            else
            { //2D: length of edge
                iv1 = elem_vert_original[icv][0];
                iv2 = elem_vert_original[icv][1];
                for (int j = 0; j < dim; j++)
                {
                    v1[j] = xyz_original[dim * iv1 + j];
                    v2[j] = xyz_original[dim * iv2 + j];
                    boundary_face_area_original[icv] = boundary_face_area_original[icv] + (v1[j] - v2[j]) * (v1[j] - v2[j]);
                }
                boundary_face_area_original[icv] = sqrt(boundary_face_area_original[icv]);
            }
        }
    }
    if (dim == 3)
    {
        for (int iv = 0; iv < vert_nb_cv.size(); iv++)
            vert_nb_cv[iv].clear();
        vert_nb_cv.clear();
        elem_normal.clear();
    }
}

void POINTS::delete_corner_edge_vertices(PARAMETERS &parameters)
{ //delete corner vertices for 2D problems and both corner and edge vertices for 3D problems
    clock_t start = clock();
    vector<double> xyz_temp, normal_temp;
    vector<bool> b_temp;
    int dim = parameters.dimension, nv1 = boundary_flag.size();
    for (int iv = 0; iv < nv1; iv++)
    {
        if (corner_edge_vertices[iv] == false)
        { //iv is a required vertex: thus copy in temporary vectors
            b_temp.push_back(boundary_flag[iv]);
            for (int i = 0; i < dim; i++)
            {
                xyz_temp.push_back(xyz[dim * iv + i]);
                normal_temp.push_back(normal[dim * iv + i]);
            }
        }
    }
    boundary_flag.clear();
    xyz.clear();
    normal.clear();
    boundary_flag.insert(boundary_flag.end(), b_temp.begin(), b_temp.end());
    xyz.insert(xyz.end(), xyz_temp.begin(), xyz_temp.end());
    normal.insert(normal.end(), normal_temp.begin(), normal_temp.end());
    b_temp.clear();
    xyz_temp.clear();
    normal_temp.clear();
    // corner_edge_vertices.clear();
    nv = boundary_flag.size();
}

void POINTS::calc_vert_normal(PARAMETERS &parameters)
{
    vector<vector<int>> elem_vert, vert_nb_cv;
    vector<bool> elem_boundary_flag; //, vert_boundary_flag;
    vector<double> elem_normal;
    read_elem_vert_complete_msh(parameters, elem_vert, elem_boundary_flag);
    calc_vert_nb_cv(parameters, vert_nb_cv, elem_vert);
    if (parameters.dimension == 2)
        calc_elem_normal_2D(elem_normal, vert_nb_cv, elem_vert, elem_boundary_flag);
    else
        calc_elem_normal_3D(elem_normal, vert_nb_cv, elem_vert, elem_boundary_flag);
    elem_boundary_flag_original = elem_boundary_flag;

    for (int iv = 0; iv < parameters.dimension * nv; iv++)
        normal.push_back(0.0);
    int nbcv, count;
    double magnitude;
    for (int iv = 0; iv < nv; iv++)
    {
        if (boundary_flag[iv])
        {
            count = 0;
            for (int i1 = 0; i1 < vert_nb_cv[iv].size(); i1++)
            {
                nbcv = vert_nb_cv[iv][i1];
                if (elem_boundary_flag[nbcv])
                {
                    for (int i2 = 0; i2 < parameters.dimension; i2++)
                        normal[parameters.dimension * iv + i2] += elem_normal[parameters.dimension * nbcv + i2];
                    count++;
                }
            }
            for (int i2 = 0; i2 < parameters.dimension; i2++)
                normal[parameters.dimension * iv + i2] /= ((double)count);
            magnitude = 0.0;
            for (int i2 = 0; i2 < parameters.dimension; i2++)
                magnitude += normal[parameters.dimension * iv + i2] * normal[parameters.dimension * iv + i2];
            magnitude = sqrt(magnitude);
            for (int i2 = 0; i2 < parameters.dimension; i2++)
                normal[parameters.dimension * iv + i2] /= magnitude;
        }
    }

    elem_boundary_flag.clear();
    elem_normal.clear();
    int ncv = elem_vert.size(), nv1 = vert_nb_cv.size();
    for (int icv = 0; icv < ncv; icv++)
        elem_vert[icv].clear();
    elem_vert.clear();
    for (int iv = 0; iv < nv1; iv++)
        vert_nb_cv[iv].clear();
    vert_nb_cv.clear();
}

void POINTS::calc_elem_normal_3D(vector<double> &elem_normal, vector<vector<int>> &vert_nb_cv, vector<vector<int>> &elem_vert, vector<bool> &elem_boundary_flag)
{
    int dim = 3;
    int nv = (int)(xyz.size() / dim), ncv = elem_vert.size();
    int iv_nb_cv_internal, iv0, iv1, iv2;
    double normal[3], direction[3], magnitude, d_temp, vec0[3], vec1[3], centroid[3];
    for (int icv = 0; icv < dim * ncv; icv++)
    {
        elem_normal.push_back(0.0);
    }
    for (int icv = 0; icv < ncv; icv++)
    {
        if (elem_boundary_flag[icv])
        {
            if (elem_vert[icv].size() < 3)
            {
                cout << "\n\nERROR from calc_elem_normal_3D boundary element icv: " << icv << " has only " << elem_vert[icv].size() << " vertices\n\n";
                throw bad_exception();
            }
            iv0 = elem_vert[icv][0];
            iv1 = elem_vert[icv][1];
            iv2 = elem_vert[icv][2];
            for (int i1 = 0; i1 < dim; i1++)
            {
                vec0[i1] = xyz[dim * iv1 + i1] - xyz[dim * iv0 + i1];
                vec1[i1] = xyz[dim * iv2 + i1] - xyz[dim * iv1 + i1];
            }
            cross_product(normal, vec0, vec1);
            iv_nb_cv_internal = -1;
            for (int i1 = 0; i1 < vert_nb_cv[iv0].size(); i1++)
            {
                if (!elem_boundary_flag[vert_nb_cv[iv0][i1]])
                { //is internal CV
                    iv_nb_cv_internal = vert_nb_cv[iv0][i1];
                    break;
                }
            }
            if (iv_nb_cv_internal == -1)
            {
                cout << "\n\nERROR from calc_elem_normal_3D boundary vertex iv0: " << iv0 << " does not have a neighboring internal element\n\n";
                cout << "\n\nERROR from calc_elem_normal_3D boundary element icv: " << icv << " does not have a neighboring internal element\n\n";
                throw bad_exception();
            }
            // iv_nb_vert_internal = -1;
            centroid[0] = 0.0;
            centroid[1] = 0.0;
            centroid[2] = 0.0;
            for (int i1 = 0; i1 < elem_vert[iv_nb_cv_internal].size(); i1++)
            {
                // if (elem_vert[iv_nb_cv_internal][i1] != iv0 && elem_vert[iv_nb_cv_internal][i1] != iv1)
                // if (!vert_boundary_flag[elem_vert[iv_nb_cv_internal][i1]])
                // { //internal vertex found
                //     iv_nb_vert_internal = elem_vert[iv_nb_cv_internal][i1];
                //     break;
                // }
                centroid[0] += xyz[dim * elem_vert[iv_nb_cv_internal][i1]];
                centroid[1] += xyz[dim * elem_vert[iv_nb_cv_internal][i1] + 1];
                centroid[2] += xyz[dim * elem_vert[iv_nb_cv_internal][i1] + 2];
            }
            // if (iv_nb_vert_internal == -1)
            // {
            //     cout << "\ncalc_elem_normal_3D internal element iv_nb_cv_internal: " << iv_nb_cv_internal << " does not have a internal vertex\n\n";
            //     throw bad_exception();
            // }
            centroid[0] = centroid[0] / ((double)(elem_vert[iv_nb_cv_internal].size()));
            centroid[1] = centroid[1] / ((double)(elem_vert[iv_nb_cv_internal].size()));
            centroid[2] = centroid[2] / ((double)(elem_vert[iv_nb_cv_internal].size()));
            // direction[0] = xyz[dim * iv1] - xyz[dim * iv_nb_vert_internal];
            // direction[1] = xyz[dim * iv1 + 1] - xyz[dim * iv_nb_vert_internal + 1];
            // direction[2] = xyz[dim * iv1 + 2] - xyz[dim * iv_nb_vert_internal + 2];
            direction[0] = xyz[dim * iv1] - centroid[0];
            direction[1] = xyz[dim * iv1 + 1] - centroid[1];
            direction[2] = xyz[dim * iv1 + 2] - centroid[2];
            if ((direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2]) < 0)
            { //make outward facing
                normal[0] = -normal[0];
                normal[1] = -normal[1];
                normal[2] = -normal[2];
            }
            magnitude = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            normal[0] = normal[0] / magnitude;
            normal[1] = normal[1] / magnitude;
            normal[2] = normal[2] / magnitude;
            elem_normal[dim * icv] = normal[0];
            elem_normal[dim * icv + 1] = normal[1];
            elem_normal[dim * icv + 2] = normal[2];
        }
    }
}

void POINTS::calc_elem_normal_2D(vector<double> &elem_normal, vector<vector<int>> &vert_nb_cv, vector<vector<int>> &elem_vert, vector<bool> &elem_boundary_flag)
{
    int dim = 2;
    int nv = (int)(xyz.size() / dim), ncv = elem_vert.size();
    int iv_nb_cv_internal, iv0, iv1;
    double normal[2], direction[2], magnitude, d_temp, centroid[2];
    for (int icv = 0; icv < dim * ncv; icv++)
    {
        elem_normal.push_back(0.0);
    }
    for (int icv = 0; icv < ncv; icv++)
    {
        if (elem_boundary_flag[icv])
        {
            iv0 = elem_vert[icv][0];
            iv1 = elem_vert[icv][1];
            normal[0] = xyz[dim * iv1 + 1] - xyz[dim * iv0 + 1];
            normal[1] = -(xyz[dim * iv1] - xyz[dim * iv0]);
            iv_nb_cv_internal = -1;
            for (int i1 = 0; i1 < vert_nb_cv[iv0].size(); i1++)
            {
                if (!elem_boundary_flag[vert_nb_cv[iv0][i1]])
                { //is internal CV
                    iv_nb_cv_internal = vert_nb_cv[iv0][i1];
                    break;
                }
            }
            if (iv_nb_cv_internal == -1)
            {
                cout << "\n\nERROR from calc_elem_normal_2D boundary vertex iv0: " << iv0 << " does not have a neighboring internal element\n\n";
                cout << "\n\nERROR from calc_elem_normal_2D boundary element icv: " << icv << " does not have a neighboring internal element\n\n";
                throw bad_exception();
            }
            // iv_nb_vert_internal = -1;
            centroid[0] = 0.0;
            centroid[1] = 0.0;
            for (int i1 = 0; i1 < elem_vert[iv_nb_cv_internal].size(); i1++)
            {
                // if (!vert_boundary_flag[elem_vert[iv_nb_cv_internal][i1]])
                // { //internal vertex found
                //     iv_nb_vert_internal = elem_vert[iv_nb_cv_internal][i1];
                //     break;
                // }
                centroid[0] += xyz[dim * elem_vert[iv_nb_cv_internal][i1]];
                centroid[1] += xyz[dim * elem_vert[iv_nb_cv_internal][i1] + 1];
            }
            centroid[0] = centroid[0] / ((double)(elem_vert[iv_nb_cv_internal].size()));
            centroid[1] = centroid[1] / ((double)(elem_vert[iv_nb_cv_internal].size()));
            // if (iv_nb_vert_internal == -1)
            // {
            //     cout << "\ncalc_elem_normal_2D internal element iv_nb_cv_internal: " << iv_nb_cv_internal << " does not have a internal vertex\n\n";
            //     throw bad_exception();
            // }
            // direction[0] = xyz[dim * iv1] - xyz[dim * iv_nb_vert_internal];
            // direction[1] = xyz[dim * iv1 + 1] - xyz[dim * iv_nb_vert_internal + 1];
            direction[0] = xyz[dim * iv1] - centroid[0];
            direction[1] = xyz[dim * iv1 + 1] - centroid[1];
            if ((direction[0] * normal[0] + direction[1] * normal[1]) < 0)
            { //make outward facing
                normal[0] = -normal[0];
                normal[1] = -normal[1];
            }
            magnitude = sqrt(normal[0] * normal[0] + normal[1] * normal[1]);
            normal[0] = normal[0] / magnitude;
            normal[1] = normal[1] / magnitude;
            elem_normal[dim * icv] = normal[0];
            elem_normal[dim * icv + 1] = normal[1];
        }
    }
}

void POINTS::read_elem_vert_complete_msh(PARAMETERS &parameters, vector<vector<int>> &elem_vert, vector<bool> &elem_boundary_flag)
{                                   //read all non-trivial elements (do not read point, do not read line for 3D)
    int dim = parameters.dimension; //problem dimension (2 or 3)
    int itemp, cv_type, ncv_full, tag, count = 0;
    double dtemp;
    char temp[50];
    vector<int> vec_temp;
    FILE *file;
    file = fopen(parameters.meshfile.c_str(), "r");
    while (true)
    {
        fscanf(file, "%s ", temp);
        if (strcmp(temp, "$Elements") == 0)
            break;
    }
    fscanf(file, "%i ", &ncv_full);
    for (int icv = 0; icv < ncv_full; icv++)
    {
        fscanf(file, "%i ", &itemp);   //cv number
        fscanf(file, "%i ", &cv_type); //cv type
        fscanf(file, "%i ", &itemp);
        fscanf(file, "%i ", &itemp);
        fscanf(file, "%i ", &tag);     //element tag
        if (cv_type == 15)             //corner vertex CV: do not read
            fscanf(file, "%*[^\n]\n"); //skip reading remaining row
        else if (cv_type == 1)
        { //2 node line CV
            if (dim == 2)
            {
                elem_boundary_flag.push_back(true);
                elem_vert.push_back(vec_temp);
                for (int i1 = 0; i1 < 2; i1++)
                {
                    fscanf(file, "%i ", &itemp); //vertex number
                    elem_vert[count].push_back(itemp - 1);
                }
                count++;
            }
            else
            {
                fscanf(file, "%*[^\n]\n"); //skip reading remaining row
            }
        }
        else if (cv_type == 2)
        { //3 node triangular CV
            elem_vert.push_back(vec_temp);
            for (int i1 = 0; i1 < 3; i1++)
            {
                fscanf(file, "%i ", &itemp); //vertex number
                elem_vert[count].push_back(itemp - 1);
            }
            if (dim == 2)
                elem_boundary_flag.push_back(false);
            else
                elem_boundary_flag.push_back(true);
            count++;
        }
        else if (cv_type == 4)
        { //4 node tetrahedral CV
            elem_vert.push_back(vec_temp);
            for (int i1 = 0; i1 < 4; i1++)
            {
                fscanf(file, "%i ", &itemp); //vertex number
                elem_vert[count].push_back(itemp - 1);
            }
            elem_boundary_flag.push_back(false);
            count++;
        }
        else
        {
            cout << "\n\nERROR from read_elem_vert_complete_msh: Unable to identify CV type: " << cv_type << "\n\n";
            throw bad_exception();
        }
    }
    fclose(file);
    elem_vert_original = elem_vert;
    nelem_original = elem_vert_original.size();
}

void POINTS::calc_vert_nb_cv(PARAMETERS &parameters, vector<vector<int>> &vert_nb_cv, vector<vector<int>> &elem_vert)
{
    clock_t start = clock();
    vector<int> vec_temp;
    int iv1, ncv = elem_vert.size();
    for (int iv = 0; iv < nv; iv++)
        vert_nb_cv.push_back(vec_temp);
    for (int icv = 0; icv < ncv; icv++)
    {
        for (int i1 = 0; i1 < elem_vert[icv].size(); i1++)
        {
            iv1 = elem_vert[icv][i1];
            vert_nb_cv[iv1].push_back(icv);
        }
    }
    for (int iv = 0; iv < nv; iv++)
        if (vert_nb_cv[iv].size() == 0)      //no nbcv
            corner_edge_vertices[iv] = true; //hanging vertex: remove it later
}

void POINTS::read_points_xyz_msh(PARAMETERS &parameters)
{
    int dim = parameters.dimension; //problem dimension (2 or 3)
    FILE *file;
    int itemp;
    double dtemp;
    char temp[50];
    file = fopen(parameters.meshfile.c_str(), "r");
    while (true)
    {
        fscanf(file, "%s ", temp);
        if (strcmp(temp, "$Nodes") == 0)
            break;
    }
    fscanf(file, "%i ", &nv);
    for (int iv = 0; iv < nv; iv++)
    {
        fscanf(file, "%i ", &itemp);  //vertex number
        fscanf(file, "%lf ", &dtemp); //x co-ordinate
        xyz.push_back(dtemp);
        fscanf(file, "%lf ", &dtemp); //y co-ordinate
        xyz.push_back(dtemp);
        fscanf(file, "%lf ", &dtemp); //z co-ordinate
        if (dim == 3)
            xyz.push_back(dtemp); //for 2D problems, Z co-ordinate should be zero and should not be stored
    }
    xyz_original = xyz;
    nv_original = nv;
}

void POINTS::read_points_flag_msh(PARAMETERS &parameters)
{
    vector<int>::iterator it;
    for (int iv = 0; iv < nv; iv++)
    { //initialize
        boundary_flag.push_back(false);
        corner_edge_vertices.push_back(false);
    }
    FILE *file;
    int itemp, ncv, cv_type, tag_int, dim = parameters.dimension;
    double dtemp;
    char temp[50];
    file = fopen(parameters.meshfile.c_str(), "r");
    while (true)
    {
        fscanf(file, "%s ", temp);
        if (strcmp(temp, "$Elements") == 0)
            break;
    }
    fscanf(file, "%i ", &ncv);
    if (dim == 2)
    { //2D problem
        for (int icv = 0; icv < ncv; icv++)
        {
            fscanf(file, "%i ", &itemp);   //cv number
            fscanf(file, "%i ", &cv_type); //cv type
            if (cv_type == 1)
            { //important 2 node line CV on the boundary
                fscanf(file, "%i ", &itemp);
                fscanf(file, "%i ", &itemp);
                fscanf(file, "%i ", &tag_int); //element tag
                for (int i = 0; i < 2; i++)
                {                                    //has 2 vertices
                    fscanf(file, "%i ", &itemp);     //vertex number
                    boundary_flag[itemp - 1] = true; //boundary vertex identified
                }
            }
            else if (cv_type == 15)
            { //corner vertex CV: log this vertex number to delete it later
                fscanf(file, "%i ", &itemp);
                fscanf(file, "%i ", &itemp);
                fscanf(file, "%i ", &tag_int); //element tag
                fscanf(file, "%i ", &itemp);   //vertex number
                corner_edge_vertices[itemp - 1] = true;
            }
            else
            {                              //not reading any other kind of CV
                fscanf(file, "%*[^\n]\n"); //skip reading remaining row
            }
        }
    }
    else
    { //3D problem
        int cv_vert_num;
        for (int icv = 0; icv < ncv; icv++)
        {
            fscanf(file, "%i ", &itemp);   //cv number
            fscanf(file, "%i ", &cv_type); //cv type
            if (cv_type == 2)
            { //important 3 node triangle CV on the boundary
                fscanf(file, "%i ", &itemp);
                fscanf(file, "%i ", &itemp);
                fscanf(file, "%i ", &tag_int); //element tag
                for (int i = 0; i < 3; i++)
                {                                    //has 3 vertices
                    fscanf(file, "%i ", &itemp);     //vertex number
                    boundary_flag[itemp - 1] = true; //boundary vertex identified
                }
            }
            else if (cv_type == 3)
            { //important 4 node quad CV on the boundary
                fscanf(file, "%i ", &itemp);
                fscanf(file, "%i ", &itemp);
                fscanf(file, "%i ", &tag_int); //element tag
                for (int i = 0; i < 4; i++)
                {                                    //has 4 vertices
                    fscanf(file, "%i ", &itemp);     //vertex number
                    boundary_flag[itemp - 1] = true; //boundary vertex identified
                }
            }
            else if (cv_type == 1)
            { //2 node line CV: log these 2 vertex numbers to delete them later
                fscanf(file, "%i ", &itemp);
                fscanf(file, "%i ", &itemp);
                fscanf(file, "%i ", &tag_int); //element tag
                fscanf(file, "%i ", &itemp);   //first vertex number
                corner_edge_vertices[itemp - 1] = true;
                fscanf(file, "%i ", &itemp); //second vertex number
                corner_edge_vertices[itemp - 1] = true;
            }
            else
            {                              //not reading any other kind of CV
                fscanf(file, "%*[^\n]\n"); //skip reading remaining row
            }
        }
    }
}