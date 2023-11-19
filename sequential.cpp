#include <bits/stdc++.h>
using namespace std;

// Function to copy the contents of one vector to another
void vector_copy_seq(vector<double> &source, vector<double> &destination) {
    if (source.size() != destination.size()) {
        cout << "Copying of incompatible vectors.\n";
        exit(0);
    }

    int n = source.size();
    for (int i = 0; i < n; i++) {
        destination[i] = source[i];
    }
}

// Function for vector addition with scaling
void add_seq(vector<double> &result, vector<double> &a, vector<double> &b, double alpha) {
    if (a.size() != b.size()) {
        cout << "Addition of incompatible vectors.\n";
        cout << "Sizes are:" << a.size() << " " << b.size() << endl;
        exit(0);
    }

    int n = a.size();
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + alpha * b[i];
    }
}

// Function to calculate the dot_seq product of two vectors
double dot_seq(vector<double> &a, vector<double> &b) {
    if (a.size() != b.size()) {
        cout << "dot_seq product of incompatible vectors.\n";
        exit(0);
    }

    int n = a.size();
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum = sum + (a[i] * b[i]);
    }
    return sum;
}

// Function for matrix-vector multiplication with a sparse matrix
void MatVecMult_seq(vector<double> &result, vector<double> &A, vector<int> &iA, vector<int> &jA,
                vector<double> &x, bool describe = false) {
    int n = x.size();
    for (int i = 0; i < n; i++) {
        result[i] = 0;
        for (int idx = iA[i]; idx < iA[i + 1]; idx++) {
            result[i] += A[idx] * x[jA[idx]];
        }
    }
    if (describe) {
        cout << "Matrix-Vector Product :\n";
        for (auto u : result)
            cout << u << ' ';
        cout << "\n\n";
    }
}

// Function to compute the transpose of a sparse matrix
void sparse_matrix_transpose_seq(vector<double> &A, vector<int> &iA, vector<int> &jA,
                             vector<double> &A_T, vector<int> &iA_T, vector<int> &jA_T, bool describe = false) {
    int i, j, k, l;

    int n = iA.size() - 1;
    int nz = A.size();

    for (i = 0; i <= n; i++)
        iA_T[i] = 0;

    for (i = 0; i < nz; i++)
        iA_T[jA[i] + 1]++;

    for (i = 0; i < n; i++)
        iA_T[i + 1] += iA_T[i];

    auto ptr = iA.begin();

    for (i = 0; i < n; i++, ptr++)
        for (j = *ptr; j < *(ptr + 1); j++) {
            k = jA[j];
            l = iA_T[k]++;
            jA_T[l] = i;
            A_T[l] = A[j];
        }

    for (i = n; i > 0; i--)
        iA_T[i] = iA_T[i - 1];

    iA_T[0] = 0;

    if (describe) {
        cout << "Sparse Matrix Transpose :\n";
        for (auto u : A_T) {
            cout << u << " ";
        }
        cout << endl;

        for (auto u : iA_T) {
            cout << u << " ";
        }
        cout << endl;

        for (auto u : jA_T) {
            cout << u << " ";
        }
        cout << "\n\n";
    }
}

// Function to check if a matrix is symmetric
bool is_symmetric_seq(vector<double> &A, vector<double> &A_T) {
    bool res = true;
    int nz = A.size();

    for (int i = 0; i < nz; i++) {
        res = res && (A[i] == A_T[i]);
    }
    return res;
}

// Conjugate Gradient Method
void sequential_Conjugate_Gradient(vector<double> &x_out, vector<double> &A, vector<int> &iA, vector<int> &jA,
                                   vector<double> &b, vector<double> init_x, int iterations, double epsilon) {
    int n = init_x.size();
    vector<double> matprod(n), x(n), r(n), p(n), rtemp(n);
    vector_copy_seq(init_x, x);
    MatVecMult_seq(matprod, A, iA, jA, x);
    add_seq(r, b, matprod, -1.0);
    vector_copy_seq(r, p);
    int it = 0;
    double alpha, beta, r_norm, rtemp_norm;

    while (it < iterations) {
        it++;
        MatVecMult_seq(matprod, A, iA, jA, p);
        r_norm = dot_seq(r, r);
        alpha = r_norm / dot_seq(p, matprod);
        add_seq(x, x, p, alpha);
        add_seq(rtemp, r, matprod, -alpha);
        rtemp_norm = dot_seq(rtemp, rtemp);
        if (rtemp_norm < epsilon)
            break;
        beta = rtemp_norm / r_norm;
        add_seq(p, rtemp, p, beta);
        vector_copy_seq(rtemp, r);
    }

    vector_copy_seq(x, x_out);
}

// Function to compute A^T * A
void A_TA_seq(vector<double> &A_T, vector<int> &iA_T, vector<int> &jA_T,
          vector<double> &B, vector<int> &iB, vector<int> &jB, bool describe = false)
{
    int n = iB.size() - 1;

    for (int i = 0; i < n + 1; i++)
        iB[i] = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int k = iA_T[i];
            int l = iA_T[j];
            double sum = 0;
            while (k < iA_T[i + 1] && l < iA_T[j + 1])
            {
                if (jA_T[k] > jA_T[l])
                    l++;
                else if (jA_T[k] < jA_T[l])
                    k++;
                else
                {
                    sum += A_T[k] * A_T[l];
                    l++;
                    k++;
                }
            }
            if (sum > 0)
            {
                B.push_back(sum);
                jB.push_back(j);
                iB[i + 1]++;
            }
        }
    }
    for (int i = 0; i < n; i++)
        iB[i + 1] += iB[i];

    if (describe)
    {
        cout << "A^T*A :\n";
        for (auto u : B)
        {
            cout << u << " ";
        }
        cout << endl;
        for (auto u : iB)
        {
            cout << u << " ";
        }
        cout << endl;
        for (auto u : jB)
        {
            cout << u << " ";
        }
        cout << "\n\n";
    }
}

// Function to check if dimensions are proper for the solver
bool check_proper_dims_seq(vector<double> &A, vector<int> &iA, vector<int> &jA, vector<double> &b)
{
    return (A.size() == jA.size()) && ((iA.size() - 1) == b.size());
}

// Main solver function
vector<double> solver_sequential(vector<double> &A, vector<int> &iA, vector<int> &jA, vector<double> &b, bool describe = false, bool assume_psd = true, const int iterations = 10, const double epsilon = 1e-5)
{
    assert(check_proper_dims_seq(A, iA, jA, b));

    int n = iA.size() - 1;
    vector<double> x_out(n);
    vector<double> x0(n, 1);

    std::chrono::steady_clock::time_point START = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point END = std::chrono::steady_clock::now();

    if (assume_psd)
    {
        START = std::chrono::steady_clock::now();
        sequential_Conjugate_Gradient(x_out, A, iA, jA, b, x0, iterations, epsilon);
        END = std::chrono::steady_clock::now();
    }
    else
    {
        vector<double> A_T(A.size());
        vector<int> A_Trow(n + 1);
        vector<int> A_Tcol(A.size());

        sparse_matrix_transpose_seq(A, iA, jA, A_T, A_Trow, A_Tcol);

        if (!is_symmetric_seq(A, A_T))
        {
            vector<double> R;
            vector<int> Rrow(n + 1);
            vector<int> Rcol;

            A_TA_seq(A_T, A_Trow, A_Tcol, R, Rrow, Rcol);

            vector<double> q(n);
            MatVecMult_seq(q, A_T, A_Trow, A_Tcol, b);

            START = std::chrono::steady_clock::now();
            sequential_Conjugate_Gradient(x_out, R, Rrow, Rcol, q, x0, iterations, epsilon);
            END = std::chrono::steady_clock::now();
        }
        else
        {
            START = std::chrono::steady_clock::now();
            sequential_Conjugate_Gradient(x_out, A, iA, jA, b, x0, iterations, epsilon);
            END = std::chrono::steady_clock::now();
        }
    }

    if (describe)
    {
        cout << "Solution x :\n";
        for (int i = 0; i < x_out.size(); i++)
            cout << x_out[i] << endl;
        std::cout << "\nExecution time (microsec) = " << (std::chrono::duration_cast<std::chrono::microseconds>(END - START).count()) << std::endl;

        vector<double> res(n);
        MatVecMult_seq(res, A, iA, jA, x_out, false);

        cout << "original b is:" << endl;
        for (auto u : b)
        {
            cout << u << " ";
        }
        cout << endl
             << "b_res is:" << endl;
        for (auto u : res)
        {
            cout << u << " ";
        }
        cout << endl;
    }

    return x_out;
}

