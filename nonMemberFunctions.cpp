#include "Matrix.h"
#include <Eigen/Dense>
#include <Eigen/SVD>

static Matrix diag_matrix(const vector<double> &diagValues)
{
    size_t size = diagValues.size();
    Matrix diagonalMatrix(size, size);
    for (size_t i = 0; i < size; ++i)
    {
        diagonalMatrix.mat_data[i * size + i] = diagValues[i];
    }

    return diagonalMatrix;
}

bool is_triangular(const Matrix &matrix)
{
    size_t r = matrix.r;
    size_t c = matrix.c;
    if (r != c)
    {
        return false;
    }
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = i + 1; j < c; ++j)
        {
            if (matrix.mat_data[i * c + j] != 0.0)
            {
                return false;
            }
        }
    }
    return true;
}

Matrix LU_Decomposition(const Matrix &matrix)
{
    size_t r = matrix.r;
    size_t c = matrix.c;
    if (r != c)
    {
        throw std::invalid_argument("Matrix must be square");
    }
    Matrix L(r, c);
    Matrix U(r, c);
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            if (i == j)
            {
                L.mat_data[i * c + j] = 1.0;
            }
            else
            {
                L.mat_data[i * c + j] = 0.0;
            }
            U.mat_data[i * c + j] = matrix.mat_data[i * c + j];
        }
    }
    for (size_t j = 0; j < c; ++j)
    {
        for (size_t i = j + 1; i < r; ++i)
        {
            double factor = U.mat_data[i * c + j] / U.mat_data[j * c + j];
            L.mat_data[i * c + j] = factor;
            for (size_t k = j; k < c; ++k)
            {
                U.mat_data[i * c + k] -= factor * U.mat_data[j * c + k];
            }
        }
    }
    return L;
}

std::pair<Matrix, Matrix> QR_factorization(Matrix &matrix)
{
    size_t r = matrix.r;
    size_t c = matrix.c;
    if (r != c)
    {
        throw std::invalid_argument("Matrix must be square");
    }
    Matrix Q(r, c);
    Matrix R(r, c);
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            if (i == j)
            {
                Q.mat_data[i * c + j] = 1.0;
            }
            else
            {
                Q.mat_data[i * c + j] = 0.0;
            }
            R.mat_data[i * c + j] = matrix.mat_data[i * c + j];
        }
    }
    for (size_t j = 0; j < c; ++j)
    {
        for (size_t i = j + 1; i < r; ++i)
        {
            double factor = R.mat_data[i * c + j] / R.mat_data[j * c + j];
            Q.mat_data[i * c + j] = factor;
            for (size_t k = j; k < c; ++k)
            {
                R.mat_data[i * c + k] -= factor * R.mat_data[j * c + k];
            }
        }
    }
    return std::make_pair(Q, R);
}

std::pair<Matrix, Matrix> eigen_decomposition(Matrix &matrix)
{
    size_t r = matrix.r;
    size_t c = matrix.c;
    Matrix A = matrix;
    Matrix Q(r, c);
    Matrix R(r, c);
    for (size_t i = 0; i < 50; ++i)
    {
        std::pair<Matrix, Matrix> qr = QR_factorization(A);
        Q = qr.first;
        R = qr.second;
        A = R * Q;
    }
    return std::make_pair(A, Q);
}


std::tuple<Matrix, Matrix, Matrix> singular_value_decomposition(Matrix &matrix)
{
    size_t r = matrix.r;
    size_t c = matrix.c;
    Matrix A = matrix;

    // Perform SVD using JacobiSVD from Eigen library
    Eigen::MatrixXd eigenMatrix(r, c);
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            eigenMatrix(i, j) = matrix.mat_data[i * c + j];
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(eigenMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Extract U, S, and V from the SVD result
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd S = svd.singularValues().asDiagonal();
    Eigen::MatrixXd V = svd.matrixV();

    // Convert U, S, and V back to your Matrix class
    Matrix U_mat(r, c);
    Matrix S_mat(r, c);
    Matrix V_mat(r, c);

    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            U_mat.mat_data[i * c + j] = U(i, j);
            S_mat.mat_data[i * c + j] = S(i, j);
            V_mat.mat_data[i * c + j] = V(i, j);
        }
    }

    return std::make_tuple(U_mat, S_mat, V_mat);
}

