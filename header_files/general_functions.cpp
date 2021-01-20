//Author: Dr. Shantanu Shahane
#include "general_functions.hpp"

Eigen::VectorXcd calc_largest_magnitude_eigenvalue(Eigen::SparseMatrix<double, Eigen::RowMajor> &matrix)
{
    int nconv;
    // cout << "\n calc_largest_magnitude_eigenvalue  \n\n";
    Spectra::SparseGenMatProd<double, Eigen::RowMajor> op_x(matrix);
    // Spectra::GenEigsSolver<double, Spectra::SMALLEST_REAL, Spectra::SparseGenMatProd<double>> eigs(&op, 1, 10);
    Spectra::GenEigsSolver<double, Spectra::LARGEST_MAGN, Spectra::SparseGenMatProd<double, Eigen::RowMajor>> eig_lr_mag(&op_x, 1, 10);
    eig_lr_mag.init();
    nconv = eig_lr_mag.compute();
    Eigen::VectorXcd evalues_lr_mag;
    if (eig_lr_mag.info() == Spectra::SUCCESSFUL)
        evalues_lr_mag = eig_lr_mag.eigenvalues();
    else
    {
        cout << "\n\nERROR from calc_largest_magnitude_eigenvalue Spectra Eigensolver not successful 1\n\n";
        throw bad_exception();
    }
    // double evalues = sqrt((evalues_lr_mag[0].real() * evalues_lr_mag[0].real()) + (evalues_lr_mag[0].imag() * evalues_lr_mag[0].imag()));
    // printf("\ncalc_largest_magnitude_eigenvalue Eigenvalue with largest magnitude: X: (%g, %g)", evalues_lr_mag[0].real(), evalues_lr_mag[0].imag());
    return evalues_lr_mag;
}

void does_file_exist(const char *fname, const char *message)
{
    ifstream f(fname);
    if (!f.good())
    { //file fname is missing: throw error and end execution
        cout << message << "\n\nERROR from does_file_exists file:" << fname << " does not exist\n\n";
        throw bad_exception();
    }
}

void check_mpi()
{
    int num_procs, myid;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (num_procs != 1)
    {
        if (myid == 0)
            cout << "\n\nERROR from check_mpi: Number of Processes currently set to: " << num_procs << " should not be greater than 1\n\n\n";
        throw bad_exception();
    }
}

void print_to_terminal(vector<bool> &a, const char *text)
{
    cout << "\nvector bool print_to_terminal called from " << text << "\nStart printing:\n\n";
    for (int i = 0; i < a.size(); i++)
    {
        cout << a[i] << "  ";
    }
    cout << "\n\nvector bool print_to_terminal called from " << text << "\nEnd printing.\n\n";
}

void print_to_terminal(vector<double> &a, const char *text)
{
    cout << "\nvector double print_to_terminal called from " << text << "\nStart printing:\n\n";
    for (int i = 0; i < a.size(); i++)
    {
        cout << a[i] << "  ";
    }
    cout << "\n\nvector double print_to_terminal called from " << text << "\nEnd printing.\n\n";
}

void print_to_terminal(Eigen::VectorXd &a, const char *text)
{
    cout << "\nEigen::VectorXd double print_to_terminal called from " << text << "\nStart printing:\n\n";
    for (int i = 0; i < a.size(); i++)
        printf("%g  ", a[i]);
    cout << "\n\nEigen::VectorXd double print_to_terminal called from " << text << "\nEnd printing.\n\n";
}

void print_to_terminal(vector<int> &a, const char *text)
{
    cout << "\nvector int print_to_terminal called from " << text << "\nStart printing:\n\n";
    for (int i = 0; i < a.size(); i++)
    {
        cout << a[i] << "  ";
    }
    cout << "\n\nvector int print_to_terminal called from " << text << "\nEnd printing.\n\n";
}

void print_to_terminal(vector<pair<int, int>> &a, const char *text)
{
    cout << "\nvector pair int print_to_terminal called from " << text << "\nStart printing:\n\n";
    for (int i = 0; i < a.size(); i++)
    {
        cout << "row " << i << ": " << a[i].first << " " << a[i].second << endl;
    }
    cout << "\n\nvector pair int print_to_terminal called from " << text << "\nEnd printing.\n\n";
}

void print_to_terminal(vector<vector<int>> &a, const char *text)
{
    cout << "\nvector int<int> print_to_terminal called from " << text << "\nStart printing:\n\n";
    for (int ir = 0; ir < a.size(); ir++)
    {
        cout << ir << " length: " << a[ir].size() << "    ";
        for (int ic = 0; ic < a[ir].size(); ic++)
        {
            cout << a[ir][ic] << "  ";
        }
        cout << endl;
    }
    cout << "\n\nvector int<int> print_to_terminal called from " << text << "\nEnd printing.\n\n";
}

void print_to_terminal(vector<double> &a, int n_row, int n_col, const char *text)
{
    int icr;
    cout << "\nvector double print_to_terminal called from " << text << "\nStart printing:\n\n";
    for (int ir = 0; ir < n_row; ir++)
    {
        cout << "row " << ir << ": ";
        for (int ic = 0; ic < n_col; ic++)
        {
            icr = ic + n_col * ir;
            cout << a[icr] << "  ";
        }
        cout << endl;
    }
    cout << "\n\nvector double print_to_terminal called from " << text << "\nEnd printing.\n\n";
}

void print_to_terminal(vector<int> &a, int n_row, int n_col, const char *text)
{
    int icr;
    cout << "\nvector int print_to_terminal called from " << text << "\nStart printing:\n\n";
    for (int ir = 0; ir < n_row; ir++)
    {
        cout << "row " << ir << ": ";
        for (int ic = 0; ic < n_col; ic++)
        {
            icr = ic + n_col * ir;
            cout << a[icr] << "  ";
        }
        cout << endl;
    }
    cout << "\n\nvector int print_to_terminal called from " << text << "\nEnd printing.\n\n";
}

void print_to_terminal(vector<bool> &a, int n_row, int n_col, const char *text)
{
    int icr;
    cout << "\nvector bool print_to_terminal called from " << text << "\nStart printing:\n\n";
    for (int ir = 0; ir < n_row; ir++)
    {
        cout << "row " << ir << ": ";
        for (int ic = 0; ic < n_col; ic++)
        {
            icr = ic + n_col * ir;
            cout << a[icr] << "  ";
        }
        cout << endl;
    }
    cout << "\n\nvector bool print_to_terminal called from " << text << "\nEnd printing.\n\n";
}

void print_to_terminal(idx_t *elem_row, idx_t *elem_col, int n_row, const char *text)
{
    cout << "\narray idx_t print_to_terminal called from " << text << "\nStart printing:\n\n";
    for (int ir = 0; ir < n_row; ir++)
    {
        cout << "row " << ir << ": ";
        for (int i1 = elem_row[ir]; i1 < elem_row[ir + 1]; i1++)
        {
            cout << elem_col[i1] << "  ";
        }
        cout << endl;
    }
    cout << "\n\narray idx_t print_to_terminal called from " << text << "\nEnd printing.\n\n";
}

void print_to_terminal(vector<int> &sp_row, vector<int> &sp_col, const char *text)
{
    int n = sp_row.size() - 1;
    cout << "\nvector bool print_to_terminal called from " << text << "\nStart printing:\n\n";
    for (int i = 0; i < n; i++)
    {
        cout << "row " << i << ": ";
        for (int j = sp_row[i]; j < sp_row[i + 1]; j++)
        {
            cout << sp_col[j] << " ";
        }
        cout << endl;
    }
    cout << "\n\nvector bool print_to_terminal called from " << text << "\nEnd printing.\n\n";
}

void print_to_terminal(Eigen::MatrixXd &A, const char *text)
{
    int nr = A.rows(), nc = A.cols();
    cout << "\nEigen::Matrix double of shape " << nr << " X " << nc << ", print_to_terminal called from " << text << "\nStart printing :\n\n";
    for (int ir = 0; ir < nr; ir++)
    {
        cout << "row " << ir << ": ";
        for (int ic = 0; ic < nc; ic++)
        {
            cout << A(ir, ic) << " ";
        }
        cout << "\n";
    }
    cout << "\nEigen::Matrix double of shape " << nr << " X " << nc << ", print_to_terminal called from " << text << "\nEnd printing.\n\n ";
}

void print_to_terminal(Eigen::MatrixXi &A, const char *text)
{
    int nr = A.rows(), nc = A.cols();
    cout << "\nEigen::Matrix int of shape " << nr << " X " << nc << ", print_to_terminal called from " << text << "\nStart printing :\n\n";
    for (int ir = 0; ir < nr; ir++)
    {
        cout << "row " << ir << ": ";
        for (int ic = 0; ic < nc; ic++)
        {
            cout << A(ir, ic) << " ";
        }
        cout << "\n";
    }
    cout << "\nEigen::Matrix int of shape " << nr << " X " << nc << ", print_to_terminal called from " << text << "\nEnd printing.\n\n ";
}

void print_to_terminal(Eigen::SparseMatrix<double, Eigen::RowMajor> &A, const char *text)
{
    cout << "\nEigen::SparseMatrix<double, Eigen::RowMajor> of shape " << A.rows() << " X " << A.cols() << " and "
         << A.nonZeros() << " nonzeros, print_to_terminal called from " << text << "\nStart printing :\n\n ";
    for (int k = 0; k < A.outerSize(); ++k)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, k); it; ++it)
            printf("%ld,%ld,%g\n", it.row(), it.col(), it.value());
    cout << "\nEigen::SparseMatrix<double, Eigen::RowMajor> of shape " << A.rows() << " X " << A.cols() << " and "
         << A.nonZeros() << " nonzeros, print_to_terminal called from " << text << "\nEnd printing :\n\n ";
}

void print_to_terminal(Eigen::SparseMatrix<int> &A, const char *text)
{
    cout << "\nEigen::SparseMatrix<int> of shape " << A.rows() << " X " << A.cols() << " and "
         << A.nonZeros() << " nonzeros, print_to_terminal called from " << text << "\nStart printing :\n\n ";
    for (int k = 0; k < A.outerSize(); ++k)
        for (Eigen::SparseMatrix<int>::InnerIterator it(A, k); it; ++it)
            printf("%ld,%ld,%i\n", it.row(), it.col(), it.value());
    cout << "\nEigen::SparseMatrix<int> of shape " << A.rows() << " X " << A.cols() << " and "
         << A.nonZeros() << " nonzeros, print_to_terminal called from " << text << "\nEnd printing :\n\n ";
}

void write_csv(Eigen::MatrixXd &A, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "w");
    int nr = A.rows(), nc = A.cols();
    for (int ir = 0; ir < nr; ir++)
    {
        fprintf(file, "%i,", ir);
        for (int ic = 0; ic < nc; ic++)
        {
            fprintf(file, "%g,", A(ir, ic));
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void write_csv(Eigen::SparseMatrix<double, Eigen::RowMajor> &A, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "w");
    fprintf(file, "No. of rows\n%ld\n", A.rows());
    fprintf(file, "No. of columns\n%ld\n", A.cols());
    fprintf(file, "No. of non-zeros\n%ld\n", A.nonZeros());
    for (int k = 0; k < A.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, k); it; ++it)
        {
            fprintf(file, "%ld,%ld,%g\n", it.row(), it.col(), it.value());
        }
    }
    fclose(file);
}

void write_csv(Eigen::SparseMatrix<int> &A, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "w");
    fprintf(file, "No. of rows\n%ld\n", A.rows());
    fprintf(file, "No. of columns\n%ld\n", A.cols());
    fprintf(file, "No. of non-zeros\n%ld\n", A.nonZeros());
    for (int k = 0; k < A.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<int>::InnerIterator it(A, k); it; ++it)
        {
            fprintf(file, "%ld,%ld,%i\n", it.row(), it.col(), it.value());
        }
    }
    fclose(file);
}

void write_csv(Eigen::VectorXd &A, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "w");
    int nr = A.size();
    for (int ir = 0; ir < nr; ir++)
    {
        fprintf(file, "%i,%g\n", ir, A(ir));
    }
    fclose(file);
}

void write_csv(vector<double> &xyz, vector<bool> &boundary_flag, Eigen::VectorXd &A_ana, Eigen::VectorXd &A_num, int dim, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "w");
    int nr = A_ana.size();
    double divide;
    if (dim == 2)
        fprintf(file, "iv,x,y,boundary,analytical,numerical,abs_error,rel_error\n");
    else
        fprintf(file, "iv,x,y,z,boundary,analytical,numerical,abs_error,rel_error\n");
    for (int iv = 0; iv < nr; iv++)
    {
        divide = fabs(A_ana[iv]);
        if (divide < 1E-10)
            divide = 0.5 * fabs(A_ana[iv] + A_num[iv]);
        if (dim == 2)
            fprintf(file, "%i,%g,%g,%i,%g,%g,%g,%g\n", iv, xyz[dim * iv], xyz[dim * iv + 1], (int)(boundary_flag[iv]), A_ana[iv], A_num[iv], fabs(A_ana[iv] - A_num[iv]), fabs(A_ana[iv] - A_num[iv]) / divide);
        else
            fprintf(file, "%i,%g,%g,%g,%i,%g,%g,%g,%g\n", iv, xyz[dim * iv], xyz[dim * iv + 1], xyz[dim * iv + 2], (int)(boundary_flag[iv]), A_ana[iv], A_num[iv], fabs(A_ana[iv] - A_num[iv]), fabs(A_ana[iv] - A_num[iv]) / divide);
    }
    fclose(file);
}

void write_csv(vector<double> &vect, int nr, int nc, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "w");
    for (int ir = 0; ir < nr; ir++)
    {
        fprintf(file, "%i,", ir);
        for (int ic = 0; ic < nc; ic++)
        {
            fprintf(file, "%g,", vect[nc * ir + ic]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void write_csv(vector<bool> &vect, int nr, int nc, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "w");
    for (int ir = 0; ir < nr; ir++)
    {
        fprintf(file, "%i,", ir);
        for (int ic = 0; ic < nc; ic++)
        {
            fprintf(file, "%i,", (int)(vect[nc * ir + ic]));
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void write_csv(double *vect, int nr, int nc, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "w");
    for (int ir = 0; ir < nr; ir++)
    {
        fprintf(file, "%i,", ir);
        for (int ic = 0; ic < nc; ic++)
        {
            fprintf(file, "%g,", vect[nc * ir + ic]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void write_csv(vector<int> &sp_row, vector<int> &sp_col, const char *file_name)
{
    FILE *file;
    int n = sp_row.size() - 1;
    file = fopen(file_name, "w");
    for (int i = 0; i < n; i++)
    {
        fprintf(file, "%i,", i);
        for (int j = sp_row[i]; j < sp_row[i + 1]; j++)
        {
            fprintf(file, "%i,", sp_col[j]);
        }
        fprintf(file, "\n ");
    }
    fclose(file);
}

void write_csv(vector<int> &sp_row, vector<double> &sp_val, const char *file_name)
{
    FILE *file;
    int n = sp_row.size() - 1;
    file = fopen(file_name, "w");
    for (int i = 0; i < n; i++)
    {
        fprintf(file, "%i,", i);
        for (int j = sp_row[i]; j < sp_row[i + 1]; j++)
        {
            fprintf(file, "%e,", sp_val[j]);
        }
        fprintf(file, "\n ");
    }
    fclose(file);
}

void write_csv(vector<vector<int>> &a, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "w");
    for (int ir = 0; ir < a.size(); ir++)
    {
        fprintf(file, "%i,", ir);
        for (int ic = 0; ic < a[ir].size(); ic++)
        {
            fprintf(file, "%i,", a[ir][ic]);
        }
        fprintf(file, "\n ");
    }
    fclose(file);
}

void write_csv(vector<vector<double>> &a, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "w");
    for (int ir = 0; ir < a.size(); ir++)
    {
        fprintf(file, "%i,", ir);
        for (int ic = 0; ic < a[ir].size(); ic++)
        {
            fprintf(file, "%g,", a[ir][ic]);
        }
        fprintf(file, "\n ");
    }
    fclose(file);
}

void write_csv(vector<tuple<int, int, double>> &a, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "w");
    int row, col;
    double value;
    for (int i = 0; i < a.size(); i++)
    {
        fprintf(file, "%i:,", i);
        row = get<0>(a[i]);
        col = get<1>(a[i]);
        value = get<2>(a[i]);
        fprintf(file, "%i,%i,%g\n", row, col, value);
    }
    fclose(file);
}

void write_csv(vector<int> &vect, int nr, int nc, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "w");
    for (int ir = 0; ir < nr; ir++)
    {
        fprintf(file, "%i,", ir);
        for (int ic = 0; ic < nc; ic++)
        {
            fprintf(file, "%i,", vect[nc * ir + ic]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void k_smallest_elements(vector<double> &k_min_a, vector<int> &k_min_a_indices, vector<double> &a, int k)
{ //a: input vector
    //k_min_a: k smallest elements in a
    //k_min_a_indices: indices of k smallest elements in a
    k_min_a.clear();
    k_min_a_indices.clear();
    int n = a.size();
    if (n <= k)
    {
        k_min_a.insert(k_min_a.end(), a.begin(), a.end());
        for (int i = 0; i < n; i++)
        {
            k_min_a_indices.push_back(i);
        }
    }
    else
    {
        k_min_a.insert(k_min_a.end(), a.begin(), a.begin() + k); //initialize with first k entries
        double local_max_val = k_min_a[0];
        int local_max_index = 0;
        for (int i = 0; i < k; i++)
        { //initialize with first k entries
            if (local_max_val < k_min_a[i])
            { //find current max in k_min_a and its local index
                local_max_val = k_min_a[i];
                local_max_index = i;
            }
            k_min_a_indices.push_back(i); //initialize with first k entries
        }

        for (int j = k; j < n; j++)
        {
            if (local_max_val > a[j])
            { //better element found in a
                k_min_a[local_max_index] = a[j];
                k_min_a_indices[local_max_index] = j;

                local_max_val = k_min_a[0];
                local_max_index = 0;
                for (int i = 0; i < k; i++)
                { //update local_max_index
                    if (local_max_val < k_min_a[i])
                    { //update current max in k_min_a and its local index
                        local_max_val = k_min_a[i];
                        local_max_index = i;
                    }
                }
            }
        }
    }

    vector<pair<double, int>> k_min_pair;
    for (int i = 0; i < k; i++)
        k_min_pair.push_back(make_pair(k_min_a[i], k_min_a_indices[i]));
    sort(k_min_pair.begin(), k_min_pair.end()); //sorts by first entry of pair
    k_min_a.clear();
    k_min_a_indices.clear();
    for (int i = 0; i < k; i++)
    {
        k_min_a.push_back(k_min_pair[i].first);
        k_min_a_indices.push_back(k_min_pair[i].second);
    }
}

vector<int> argsort(const vector<double> &v)
{ //reference: https://stackoverflow.com/a/12399290
    //get indices for sorting vector v
    vector<int> idx(v.size(), 0.0);
    iota(idx.begin(), idx.end(), 0); //initialize original index locations

    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; }); //sort indices based on comparing values in v

    return idx;
}

vector<double> calc_crowd_distance(vector<double> &x, vector<double> &y, vector<double> &z, int dim)
{ //uses crowd distancing logic of NSGA-II: refer {Deb, Kalyanmoy, et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II." IEEE transactions on evolutionary computation 6.2 (2002): 182-197.}
    int n = x.size();
    vector<double> crowd_distance(n, 0.0);
    vector<int> index;
    double scale;

    index = argsort(x);
    scale = x[index[n - 1]] - x[index[0]];
    // print_to_terminal(index, "calc_crowd_distance x index");
    crowd_distance[index[0]] = 10000.0;     //infinity at extremes
    crowd_distance[index[n - 1]] = 10000.0; //infinity at extremes
    for (int i = 1; i < n - 1; i++)
    {
        crowd_distance[index[i]] = crowd_distance[index[i]] + ((x[index[i + 1]] - x[index[i - 1]]) / scale);
    }

    index = argsort(y);
    scale = y[index[n - 1]] - y[index[0]];
    // print_to_terminal(index, "calc_crowd_distance y index");
    crowd_distance[index[0]] = 10000.0;     //infinity at extremes
    crowd_distance[index[n - 1]] = 10000.0; //infinity at extremes
    for (int i = 1; i < n - 1; i++)
    {
        crowd_distance[index[i]] = crowd_distance[index[i]] + ((y[index[i + 1]] - y[index[i - 1]]) / scale);
    }

    if (dim == 3)
    {
        index = argsort(y);
        scale = z[index[n - 1]] - z[index[0]];
        // print_to_terminal(index, "calc_crowd_distance z index");
        crowd_distance[index[0]] = 10000.0;     //infinity at extremes
        crowd_distance[index[n - 1]] = 10000.0; //infinity at extremes
        for (int i = 1; i < n - 1; i++)
        {
            crowd_distance[index[i]] = crowd_distance[index[i]] + ((z[index[i + 1]] - z[index[i - 1]]) / scale);
        }
    }

    return crowd_distance;
}

double max_abs(Eigen::VectorXd &a)
{
    double max = fabs(a(0));
    for (int i = 0; i < a.size(); i++)
    {
        if (max < fabs(a(i)))
        {
            max = fabs(a(i));
        }
    }
    return max;
}

void cuthill_mckee_ordering(vector<vector<int>> &adjacency, vector<int> &order)
{
    vector<int> degree, new_numbering;
    vector<bool> visited_flag;
    int nv = adjacency.size(), iv_1, iv_2;
    for (int i = 0; i < nv; i++)
    {
        degree.push_back(adjacency[i].size());
        visited_flag.push_back(false);
    }
    new_numbering.clear();
    int iv_min_degree = 0;
    int min_degree = degree[0];
    for (int i = 0; i < nv; i++)
    {
        if (min_degree > degree[i])
        {
            min_degree = degree[i];
            iv_min_degree = i;
        }
    }
    new_numbering.push_back(iv_min_degree);
    visited_flag[iv_min_degree] = true;
    // cout << "\nreverse_cuthill_mckee_ordering iv_min_degree: " << iv_min_degree << ", min_degree: " << min_degree << "\n\n";
    vector<pair<int, int>> new_set;
    for (int index = 0; index < nv; index++)
    {
        if (new_numbering.size() == nv)
        {
            break;
        }
        iv_1 = new_numbering[index];
        // cout << "\nreverse_cuthill_mckee_ordering iv_1: " << iv_1 << ", index: " << index << "\n\n";
        // print_to_terminal(adjacency[iv_1], "reverse_cuthill_mckee_ordering adjacency[iv_1]");
        for (int i1 = 0; i1 < adjacency[iv_1].size(); i1++)
        {
            iv_2 = adjacency[iv_1][i1];
            if (!visited_flag[iv_2])
            {
                visited_flag[iv_2] = true;
                new_set.push_back(make_pair(degree[iv_2], iv_2));
            }
            sort(new_set.begin(), new_set.end());
        }
        for (int i1 = 0; i1 < new_set.size(); i1++)
        {
            new_numbering.push_back(new_set[i1].second);
        }
        // print_to_terminal(new_set, "reverse_cuthill_mckee_ordering new_set");
        new_set.clear();
    }
    visited_flag.clear();
    degree.clear();
    for (int iv = 0; iv < nv; iv++)
    {
        order.push_back(-1);
    }
    for (int iv = 0; iv < nv; iv++)
    {
        order[new_numbering[iv]] = iv;
    }
    new_numbering.clear();
}

void reverse_cuthill_mckee_ordering(vector<vector<int>> &adjacency, vector<int> &order)
{
    cuthill_mckee_ordering(adjacency, order);
    int nv = adjacency.size();
    for (int iv = 0; iv < nv; iv++)
    {
        order[iv] = nv - order[iv] - 1;
    }
}

double vector_norm(vector<double> &a, int norm_type)
{
    double norm1 = 0.0;
    if (norm_type == 1)
    { //L1 norm
        for (int i = 0; i < a.size(); i++)
        {
            norm1 += fabs(a[i]);
        }
    }
    else if (norm_type == 2)
    { //L2 norm
        for (int i = 0; i < a.size(); i++)
        {
            norm1 += (a[i] * a[i]);
        }
        norm1 = sqrt(norm1);
    }
    else
    {
        cout << "\n\nERROR from vector_norm undefined norm_type: " << norm_type << "\n\n";
        throw bad_exception();
    }
    return norm1;
}

double vector_norm(double *a, int size, int norm_type)
{
    double norm1 = 0.0;
    if (norm_type == 1)
    { //L1 norm
        for (int i = 0; i < size; i++)
        {
            norm1 += fabs(a[i]);
        }
    }
    else if (norm_type == 2)
    { //L2 norm
        for (int i = 0; i < size; i++)
        {
            norm1 += (a[i] * a[i]);
        }
        norm1 = sqrt(norm1);
    }
    else
    {
        cout << "\n\nERROR from vector_norm undefined norm_type: " << norm_type << "\n\n";
        throw bad_exception();
    }
    return norm1;
}

void cross_product(double *result, double *u, double *v)
{ //result = u X v
    result[0] = u[1] * v[2] - u[2] * v[1];
    result[1] = -u[0] * v[2] + u[2] * v[0];
    result[2] = u[0] * v[1] - u[1] * v[0];
}

void calc_max_l1_error(vector<double> &a1, vector<double> &a2, double &max_err, double &l1_err)
{
    l1_err = 0.0, max_err = 0.0;
    if (a1.size() != a2.size())
    {
        cout << "\n\nERROR from calc_max_l1_error a1.size(): " << a1.size() << ", a2.size(): " << a2.size() << "\n\n";
        throw bad_exception();
    }
    int size = a1.size();
    for (int iv = 0; iv < size; iv++)
    {
        l1_err += fabs(a1[iv] - a2[iv]);
        if (max_err < fabs(a1[iv] - a2[iv]))
            max_err = fabs(a1[iv] - a2[iv]);
    }
    l1_err = l1_err / size;
}

void calc_max_l1_relative_error(vector<double> &ana_val, vector<double> &num_val, double &max_err, double &l1_err)
{ //analytical: ana_val, numerical: num_val
    calc_max_l1_error(ana_val, num_val, max_err, l1_err);
    double max_val = ana_val[0];
    for (int iv = 0; iv < ana_val.size(); iv++)
        if (max_val < fabs(ana_val[iv]))
            max_val = fabs(ana_val[iv]);
    max_err = max_err / max_val;
    l1_err = l1_err / max_val;
}

void calc_max_l1_error(Eigen::VectorXd &a1, Eigen::VectorXd &a2, double &max_err, double &l1_err)
{
    l1_err = 0.0, max_err = 0.0;
    if (a1.size() != a2.size())
    {
        cout << "\n\nERROR from calc_max_l1_error a1.size(): " << a1.size() << ", a2.size(): " << a2.size() << "\n\n";
        throw bad_exception();
    }
    int size = a1.size();
    for (int iv = 0; iv < size; iv++)
    {
        l1_err += fabs(a1[iv] - a2[iv]);
        if (max_err < fabs(a1[iv] - a2[iv]))
            max_err = fabs(a1[iv] - a2[iv]);
    }
    l1_err = l1_err / size;
}

void calc_max_l1_relative_error(Eigen::VectorXd &ana_val, Eigen::VectorXd &num_val, double &max_err, double &l1_err)
{
    calc_max_l1_error(ana_val, num_val, max_err, l1_err);
    double max_val = ana_val[0];
    for (int iv = 0; iv < ana_val.size(); iv++)
        if (max_val < fabs(ana_val[iv]))
            max_val = fabs(ana_val[iv]);
    max_err = max_err / max_val;
    l1_err = l1_err / max_val;
}

void calc_max_l1_error(Eigen::VectorXd &a1, Eigen::VectorXd &a2, double &max_err_boundary, double &l1_err_boundary, double &max_err_internal, double &l1_err_internal, vector<bool> &boundary_flag)
{
    l1_err_boundary = 0.0, max_err_boundary = 0.0;
    l1_err_internal = 0.0, max_err_internal = 0.0;
    if (a1.size() != a2.size())
    {
        cout << "\n\nERROR from calc_max_l1_error a1.size(): " << a1.size() << ", a2.size(): " << a2.size() << "\n\n";
        throw bad_exception();
    }
    int size = a1.size(), boundary_size = 0, internal_size = 0;
    for (int iv = 0; iv < size; iv++)
    {
        if (boundary_flag[iv])
        {
            boundary_size++;
            l1_err_boundary += fabs(a1[iv] - a2[iv]);
            if (max_err_boundary < fabs(a1[iv] - a2[iv]))
                max_err_boundary = fabs(a1[iv] - a2[iv]);
        }
        else
        {
            internal_size++;
            l1_err_internal += fabs(a1[iv] - a2[iv]);
            if (max_err_internal < fabs(a1[iv] - a2[iv]))
                max_err_internal = fabs(a1[iv] - a2[iv]);
        }
    }
    l1_err_boundary = l1_err_boundary / boundary_size;
    l1_err_internal = l1_err_internal / internal_size;
}

void calc_max_l1_relative_error(Eigen::VectorXd &ana_val, Eigen::VectorXd &num_val, double &max_err_boundary, double &l1_err_boundary, double &max_err_internal, double &l1_err_internal, vector<bool> &boundary_flag)
{
    calc_max_l1_error(ana_val, num_val, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, boundary_flag);
    double max_val = ana_val[0];
    for (int iv = 0; iv < ana_val.size(); iv++)
        if (max_val < fabs(ana_val[iv]))
            max_val = fabs(ana_val[iv]);
    max_err_boundary = max_err_boundary / max_val;
    l1_err_boundary = l1_err_boundary / max_val;
    max_err_internal = max_err_internal / max_val;
    l1_err_internal = l1_err_internal / max_val;
}

void gauss_siedel_eigen(Eigen::SparseMatrix<double, Eigen::RowMajor> &matrix, Eigen::VectorXd &source, Eigen::VectorXd &field_old, int num_iter, double omega)
{ //NOT YET TESTED
    // Eigen::VectorXd field_new = field_old;
    // if (matrix.rows() < 110)
    // {
    // Eigen::MatrixXd d_matrix = Eigen::MatrixXd(matrix);
    // print_to_terminal(d_matrix, "gauss_siedel_eigen");
    // }
    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix_lowdiag = matrix.triangularView<Eigen::Lower>();    //lower+diag
    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix_up = matrix.triangularView<Eigen::StrictlyUpper>(); //upper
    Eigen::SparseMatrix<double, Eigen::RowMajor> rhs_matrix = (matrix_up * omega) + (matrix_lowdiag * (1.0 - omega));
    // if (matrix.rows() < 110)
    // {
    // Eigen::MatrixXd d_matrix = Eigen::MatrixXd(matrix);
    // print_to_terminal(matrix_lowdiag, "gauss_siedel_eigen matrix_lowdiag");
    // print_to_terminal(matrix_up, "gauss_siedel_eigen matrix_up");
    // cout << "gauss_siedel_eigen:\n"
    //  << matrix_lowdiag << "\n\n";
    // }
    cout << "\ngauss_siedel_eigen " << matrix.nonZeros() << ", " << matrix_lowdiag.nonZeros() << ", " << matrix_up.nonZeros() << ", " << rhs_matrix.nonZeros() << "\n\n";
    for (int iter = 0; iter < num_iter; iter++)
        field_old = matrix_lowdiag.triangularView<Eigen::Lower>().solve((source * omega) + (rhs_matrix * field_old));
}

Eigen::SparseMatrix<double, Eigen::RowMajor> convert_csc_to_csr_eigen(Eigen::SparseMatrix<double, Eigen::ColMajor> &matrix)
{
    vector<Eigen::Triplet<double>> triplet;
    for (int k = 0; k < matrix.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it)
            triplet.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
    Eigen::SparseMatrix<double, Eigen::RowMajor> output;
    output.resize(matrix.rows(), matrix.cols());
    output.setFromTriplets(triplet.begin(), triplet.end());
    output.makeCompressed();
    triplet.clear();
    return output;
}

Eigen::SparseMatrix<double, Eigen::ColMajor> convert_csr_to_csc_eigen(Eigen::SparseMatrix<double, Eigen::RowMajor> &matrix)
{
    vector<Eigen::Triplet<double>> triplet;
    for (int k = 0; k < matrix.outerSize(); ++k)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(matrix, k); it; ++it)
            triplet.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
    Eigen::SparseMatrix<double, Eigen::ColMajor> output;
    output.resize(matrix.rows(), matrix.cols());
    output.setFromTriplets(triplet.begin(), triplet.end());
    output.makeCompressed();
    triplet.clear();
    return output;
}