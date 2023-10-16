#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
using namespace std;

class Matrix
{
private:
    int nb_row, nb_col;
    vector<double> mat_data;

public:
    // Constructors
    Matrix();
    Matrix(int, int);
    Matrix(std::initializer_list<std::initializer_list<double>>);
    Matrix(const std::vector<std::vector<double>> &);
    Matrix(const std::string &);

    // Overloaded functions
    void operator+(const Matrix &m);
    void operator+=(const Matrix &m);
    void operator*(const Matrix &m);
    void operator*=(const Matrix &m);
    void operator-(const Matrix &m);
    void operator-=(const Matrix &m);
    void operator/(const Matrix &m);
    void operator==(const Matrix &m);
    void operator!=(const Matrix &m);

    // Other member functions
    // Member function to get a column
    vector<double> column(int col);
    // Member function to get a row
    vector<double> row(int row);
    // Member function sub_matrix returns a submatrix of a matrix.
    Matrix sub_matrix(int row, int col, int height, int width);
    void shape();
    void reshape(int row, int col);
    void transpose();
    void add_row(vector<double> v);
    void add_col(vector<double> v);
    void remove_column(int col);
    void reorder_column(int col, int new_col);
    void sort_matrix();
    void T();
    friend void transpose(Matrix &m);

    void id();
    double sum();
    double avg();
    void head();
    void print(); // Print formatted form as string.

    void to_csv(string filename);
};

// Non-member functions

Matrix diag_matrix(const Matrix &m);
bool is_triangular(const Matrix &m);
void LU_decomposition(const Matrix &m);
void QR_decomposition(const Matrix &m);
void eigen_decomposition(const Matrix &m);
void svd_decomposition(const Matrix &m);

Matrix::Matrix() : nb_row(0), nb_col(0), mat_data() {}

Matrix::Matrix(int rows, int cols) : nb_row(rows), nb_col(cols), mat_data(rows * cols, 0.0) {}

// Constructor to create a matrix from an initializer list
Matrix::Matrix(std::initializer_list<std::initializer_list<double>> init_list)
{
    nb_row = init_list.size();
    nb_col = init_list.begin()->size();
    mat_data.reserve(nb_row * nb_col);

    for (const auto &row_list : init_list)
    {
        for (double val : row_list)
        {
            mat_data.push_back(val);
        }
    }
}

// Constructor to create a matrix from a vector of vectors
Matrix::Matrix(const std::vector<std::vector<double>> &data)
{
    nb_row = data.size();
    nb_col = (nb_row > 0) ? data[0].size() : 0;
    mat_data.reserve(nb_row * nb_col);

    for (const auto &row_vec : data)
    {
        for (double val : row_vec)
        {
            mat_data.push_back(val);
        }
    }
}

Matrix::Matrix(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Error opening file");
    }

    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(file, line))
    {
        std::vector<double> row_data;
        std::istringstream iss(line);
        double val;
        while (iss >> val)
        {
            row_data.push_back(val);
            if (iss.peek() == ',')
            {
                iss.ignore();
            }
        }
        data.push_back(row_data);
    }

    nb_row = data.size();
    nb_col = (nb_row > 0) ? data[0].size() : 0;
    mat_data.reserve(nb_row * nb_col);

    for (const auto &row_vec : data)
    {
        for (double val : row_vec)
        {
            mat_data.push_back(val);
        }
    }
}

void Matrix::operator+(const Matrix &m)
{
    if (nb_row != m.nb_row || nb_col != m.nb_col)
    {
        throw std::runtime_error("Error: Matrix dimensions must agree.");
    }
    else
    {
        for (int i = 0; i < nb_row; i++)
        {
            for (int j = 0; j < nb_col; j++)
            {
                mat_data[i * nb_col + j] += m.mat_data[i * nb_col + j];
            }
        }
    }
}

void Matrix::operator+=(const Matrix &m)
{
    if (nb_row != m.nb_row || nb_col != m.nb_col)
    {
        throw std::runtime_error("Error: Matrix dimensions must agree.");
    }
    else
    {
        for (int i = 0; i < nb_row; i++)
        {
            for (int j = 0; j < nb_col; j++)
            {
                mat_data[i * nb_col + j] += m.mat_data[i * nb_col + j];
            }
        }
    }
}

void Matrix::operator*(const Matrix &m)
{
    if (nb_col != m.nb_row)
    {
        throw std::runtime_error("Error: Matrix dimensions must agree.");
    }
    else
    {
        vector<double> new_mat_data;
        new_mat_data.reserve(nb_row * m.nb_col);
        for (int i = 0; i < nb_row; i++)
        {
            for (int j = 0; j < m.nb_col; j++)
            {
                double sum = 0;
                for (int k = 0; k < nb_col; k++)
                {
                    sum += mat_data[i * nb_col + k] * m.mat_data[k * m.nb_col + j];
                }
                new_mat_data.push_back(sum);
            }
        }
        nb_col = m.nb_col;
        mat_data = new_mat_data;
    }
}

void Matrix::print()
{
    std::cout << "[";
    for (int i = 0; i < nb_row; i++)
    {
        std::cout << "[";
        for (int j = 0; j < nb_col; j++)
        {
            std::cout << mat_data[i * nb_col + j];
            if (j < nb_col - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i < nb_row - 1)
        {
            std::cout << std::endl
                      << "  ";
        }
    }
    std::cout << "]" << std::endl;
}

void Matrix::shape()
{
    cout << "Rows: " << nb_row << endl
         << "Columns: " << nb_col << endl;
}

void Matrix::reshape(int row, int col)
{
    if (row * col != nb_row * nb_col)
    {
        throw std::runtime_error("Error: Matrix dimensions must agree.");
    }
    else
    {
        nb_row = row;
        nb_col = col;

        vector<double> new_mat_data;
        new_mat_data.reserve(nb_row * nb_col);
        for (int i = 0; i < nb_row; i++)
        {
            for (int j = 0; j < nb_col; j++)
            {
                new_mat_data.push_back(mat_data[i * nb_col + j]);
            }
        }
        mat_data = new_mat_data;
    }
}
