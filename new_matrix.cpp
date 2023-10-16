// #ifndef MATRIX_H
// #define MATRIX_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <initializer_list>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
using namespace std;

class Matrix
{
private:
    size_t r;
    size_t c;
    vector<double> mat_data;

public:
    // Constructors
    Matrix(size_t rows, size_t cols);
    Matrix(const vector<vector<double>> &data);
    Matrix(initializer_list<initializer_list<double>> data);
    Matrix(size_t rows, size_t cols, bool useUniform, double mean = 0.0, double stddev = 1.0);
    Matrix(const string &filename);

    // Overloaded operators
    Matrix operator*(const Matrix &other) const;
    Matrix &operator*=(const Matrix &other);
    Matrix operator+(const Matrix &other) const;
    Matrix &operator+=(const Matrix &other);
    Matrix operator-(const Matrix &other) const;
    Matrix &operator-=(const Matrix &other);
    Matrix operator/(double scalar) const;
    bool operator==(const Matrix &other) const;
    bool operator!=(const Matrix &other) const;

    // Member functions
    vector<double> column(size_t col) const;
    vector<double> row(size_t row) const;
    Matrix sub_matrix(size_t start_row, size_t start_col, size_t end_row, size_t end_col) const;
    pair<size_t, size_t> shape() const;
    void reshape(size_t new_rows, size_t new_cols);
    void add_row(const vector<double> &new_row);
    void add_column(const vector<double> &new_col);
    void remove_column(size_t x);
    void remove_row(size_t x);
    void reorder_column(size_t col, const vector<size_t> &order);
    void sort_matrix(size_t column);
    Matrix T() const;
    friend Matrix transpose(const Matrix &matrix);
    Matrix Id(size_t size);
    double sum() const;
    void head() const;
    void print(const string &additionalString, double additionalValue) const;
    double avg() const;
    void to_csv(const string &filename) const;

    // Non-member functions
    static Matrix diag_matrix(const vector<double> &diag);
    bool is_triangular() const;
    void setDiagonal(const vector<double> &diagValues);
};

Matrix::Matrix(size_t rows, size_t cols)
{
    r = rows;
    c = cols;
}

Matrix::Matrix(const vector<vector<double>> &data)
{
    r = data.size();
    if (r > 0)
    {
        c = data[0].size();
        mat_data.reserve(r * c);
        for (size_t i = 0; i < data.size(); ++i)
        {
            const vector<double> &row = data[i];
            if (row.size() != c)
            {
                throw invalid_argument("Input matrix rows have different sizes");
            }
            mat_data.insert(mat_data.end(), row.begin(), row.end());
        }
    }
    else
        c = 0;
}

Matrix::Matrix(initializer_list<initializer_list<double>> data)
{
    r = data.size();
    if (r > 0)
    {
        c = data.begin()->size();
        mat_data.reserve(r * c);
        for (const auto &row : data)
        {
            if (row.size() != c)
            {
                throw invalid_argument("Input matrix rows have different sizes");
            }
            mat_data.insert(mat_data.end(), row.begin(), row.end());
        }
    }
    else
    {
        c = 0;
    }
}

Matrix::Matrix(size_t rows, size_t cols, bool useUniform, double mean, double stddev)
{
    r = rows;
    c = cols;

    random_device rd;
    mt19937 generator(rd());

    if (useUniform)
    {
        uniform_real_distribution<double> uniformDistribution(0.0, 1.0);

        mat_data.reserve(r * c);
        for (size_t i = 0; i < r * c; ++i)
        {
            mat_data.push_back(uniformDistribution(generator));
        }
    }
    else
    {
        normal_distribution<double> normalDistribution(mean, stddev);

        mat_data.reserve(r * c);
        for (size_t i = 0; i < r * c; ++i)
        {
            mat_data.push_back(normalDistribution(generator));
        }
    }
}

Matrix::Matrix(const string &filename)
{
    ifstream file(filename); // Open the file

    if (!file.is_open())
    {
        throw runtime_error("Failed to open CSV file");
    }

    string line;
    while (getline(file, line))
    {
        istringstream iss(line);
        double value;
        vector<double> rowValues;

        while (iss >> value)
        {
            rowValues.push_back(value);
            if (iss.peek() == ',')
            {
                iss.ignore();
            }
        }

        if (!rowValues.empty())
        {
            if (c == 0)
            {
                c = rowValues.size();
            }
            else if (rowValues.size() != c)
            {
                // Print the problematic row for debugging
                cout << "Problematic Row: " << line << endl;
                throw runtime_error("CSV rows have different sizes");
            }
            mat_data.insert(mat_data.end(), rowValues.begin(), rowValues.end());
            r++;
        }
    }
}


Matrix Matrix::operator*(const Matrix &other) const
{
    if (c != other.r)
    {
        throw invalid_argument("Matrix dimensions are not compatible");
    }
    Matrix result(r, other.c);
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < other.c; ++j)
        {
            double sum = 0.0;
            for (size_t k = 0; k < c; ++k)
            {
                sum += mat_data[i * c + k] * other.mat_data[k * other.c + j];
            }
            result.mat_data[i * other.c + j] = sum;
        }
    }
    return result;
}

Matrix &Matrix::operator*=(const Matrix &other)
{
    *this = *this * other;
    return *this;
}

Matrix Matrix::operator+(const Matrix &other) const
{
    if (r != other.r || c != other.c)
    {
        throw invalid_argument("Matrix dimensions are not compatible");
    }
    Matrix result(r, c);
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            result.mat_data[i * c + j] = mat_data[i * c + j] + other.mat_data[i * c + j];
        }
    }
    return result;
}

Matrix &Matrix::operator+=(const Matrix &other)
{
    *this = *this + other;
    return *this;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    if (r != other.r || c != other.c)
    {
        throw invalid_argument("Matrix dimensions are not compatible");
    }
    Matrix result(r, c);
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            result.mat_data[i * c + j] = mat_data[i * c + j] - other.mat_data[i * c + j];
        }
    }
    return result;
}

Matrix &Matrix::operator-=(const Matrix &other)
{
    *this = *this - other;
    return *this;
}

Matrix Matrix::operator/(double scalar) const
{
    if (scalar == 0.0)
    {
        throw runtime_error("Division by zero");
    }
    Matrix result(r, c);
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            result.mat_data[i * c + j] = mat_data[i * c + j] / scalar;
        }
    }
    return result;
}

bool Matrix::operator==(const Matrix &other) const
{
    if (r != other.r || c != other.c)
    {
        return false;
    }
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            if (mat_data[i * c + j] != other.mat_data[i * c + j])
            {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::operator!=(const Matrix &other) const
{
    return !(*this == other);
}

vector<double> Matrix::column(size_t col) const
{
    if (col >= c)
    {
        throw out_of_range("Column index out of range");
    }
    vector<double> columnData;
    columnData.reserve(r);
    for (size_t i = 0; i < r; ++i)
    {
        columnData.push_back(mat_data[i * c + col]);
    }
    return columnData;
}

vector<double> Matrix::row(size_t row) const
{
    if (row >= r)
    {
        throw out_of_range("Row index out of range");
    }
    vector<double> rowData;
    rowData.reserve(c);
    for (size_t i = 0; i < c; ++i)
    {
        rowData.push_back(mat_data[row * c + i]);
    }
    return rowData;
}

Matrix Matrix::sub_matrix(size_t start_row, size_t start_col, size_t end_row, size_t end_col) const
{
    if (start_row >= r || start_col >= c || end_row >= r || end_col >= c)
    {
        throw out_of_range("Submatrix indices are out of range");
    }

    size_t sub_rows = end_row - start_row + 1;
    size_t sub_cols = end_col - start_col + 1;
    Matrix submatrix(sub_rows, sub_cols);
    for (size_t i = start_row; i <= end_row; ++i)
    {
        for (size_t j = start_col; j <= end_col; ++j)
        {
            submatrix.mat_data.push_back(mat_data[i * c + j]);
        }
    }

    return submatrix;
}

pair<size_t, size_t> Matrix::shape() const
{
    return make_pair(r, c);
}

void Matrix::reshape(size_t new_rows, size_t new_cols)
{
    if (r * c != new_rows * new_cols)
    {
        throw invalid_argument("New shape must be compatible with the original shape");
    }
    r = new_rows;
    c = new_cols;
}

void Matrix::add_row(const vector<double> &new_row)
{
    if (new_row.size() != c)
    {
        throw invalid_argument("New row size does not match matrix's number of columns");
    }
    mat_data.insert(mat_data.end(), new_row.begin(), new_row.end());
    r++;
}

void Matrix::add_column(const vector<double> &new_col)
{
    if (new_col.size() != r)
    {
        throw invalid_argument("New column size does not match matrix's number of rows");
    }
    for (size_t i = 0; i < r; ++i)
    {
        mat_data.insert(mat_data.begin() + (i * (c + 1)), new_col[i]);
    }
    c++;
}

void Matrix::remove_column(size_t x)
{
    if (x >= c)
    {
        throw out_of_range("Column index out of range");
    }
    for (size_t i = 0; i < r; ++i)
    {
        mat_data.erase(mat_data.begin() + (i * c) + x);
    }
    c--;
}

void Matrix::remove_row(size_t x)
{
    if (x >= r)
    {
        throw out_of_range("Row index out of range");
    }
    mat_data.erase(mat_data.begin() + (x * c), mat_data.begin() + ((x + 1) * c));
    r--;
}

void Matrix::reorder_column(size_t col, const vector<size_t> &order)
{
    if (col >= c)
    {
        throw out_of_range("Column index out of range");
    }
    if (order.size() != r)
    {
        throw invalid_argument("Order vector size does not match matrix's number of rows");
    }
    vector<double> reorderedColumn;
    reorderedColumn.reserve(r);
    for (size_t i = 0; i < r; ++i)
    {
        reorderedColumn.push_back(mat_data[i * c + col]);
    }
    for (size_t i = 0; i < r; ++i)
    {
        mat_data[i * c + col] = reorderedColumn[order[i]];
    }
}

void Matrix::sort_matrix(size_t column)
{
    if (column >= c)
    {
        throw out_of_range("Column index out of range");
    }
    vector<pair<double, size_t>> columnDataWithIndices;
    columnDataWithIndices.reserve(r);
    for (size_t i = 0; i < r; ++i)
    {
        columnDataWithIndices.emplace_back(mat_data[i * c + column], i);
    }
    sort(columnDataWithIndices.begin(), columnDataWithIndices.end(),
         [](const auto &a, const auto &b)
         { return a.first < b.first; });
    Matrix sortedMatrix(r, c);
    for (size_t i = 0; i < r; ++i)
    {
        size_t sortedRowIndex = columnDataWithIndices[i].second;
        for (size_t j = 0; j < c; ++j)
        {
            sortedMatrix.mat_data[i * c + j] = mat_data[sortedRowIndex * c + j];
        }
    }

    // Copy the sorted matrix back to the original matrix
    mat_data = sortedMatrix.mat_data;
}

Matrix Matrix::T() const
{
    Matrix transposed(c, r); // Create a new matrix with swapped dimensions
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            transposed.mat_data[j * r + i] = mat_data[i * c + j]; // Transpose the elements
        }
    }
    return transposed;
}

Matrix transpose(const Matrix &matrix)
{
    Matrix transposed(matrix.c, matrix.r); // Create a new matrix with swapped dimensions
    for (size_t i = 0; i < matrix.r; ++i)
    {
        for (size_t j = 0; j < matrix.c; ++j)
        {
            transposed.mat_data[j * matrix.r + i] = matrix.mat_data[i * matrix.c + j]; // Transpose the elements
        }
    }
    return transposed;
}

Matrix Matrix::Id(size_t size)
{
    Matrix identity(size, size); // Create a square matrix of the given size
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            if (i == j)
            {
                identity.mat_data[i * size + j] = 1.0; // Set diagonal elements to 1
            }
            else
            {
                identity.mat_data[i * size + j] = 0.0; // Set non-diagonal elements to 0
            }
        }
    }
    return identity;
}

double Matrix::sum() const
{
    double totalSum = 0.0;
    for (const double &value : mat_data)
    {
        totalSum += value;
    }
    return totalSum;
}

double Matrix::avg() const
{
    if (mat_data.empty())
    {
        throw runtime_error("Matrix is empty, cannot calculate average.");
    }

    double totalSum = sum();
    size_t totalElements = mat_data.size();
    return totalSum / static_cast<double>(totalElements);
}

void Matrix::head() const
{
    if (r == 0 || c == 0)
    {
        cout << "Matrix is empty." << endl;
        return;
    }
    cout << "First row of the matrix:" << endl;
    for (size_t j = 0; j < c; ++j)
    {
        cout << mat_data[j] << " ";
    }
    cout << endl;
}

void Matrix::print(const string &additionalString, double additionalValue) const
{
    cout << "Matrix Contents:" << endl;
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            cout << setw(8) << mat_data[i * c + j] << " "; // Adjust the width as needed
        }
        cout << endl;
    }

    cout << "Additional String: " << additionalString << endl;
    cout << "Additional Value: " << additionalValue << endl;
}

void Matrix::to_csv(const string &filename) const
{
    ofstream file(filename); // Open the file for writing

    if (!file.is_open())
    {
        throw runtime_error("Failed to open CSV file for writing");
    }

    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            file << mat_data[i * c + j];
            if (j < c - 1)
            {
                file << ','; // Add a comma between values, except for the last value in a row
            }
        }
        file << endl; // Start a new line after each row
    }

    file.close(); // Close the file when done
}

Matrix Matrix::diag_matrix(const vector<double> &diag)
{
    size_t size = diag.size();
    Matrix diagonalMatrix(size, size);
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            if (i == j)
            {
                diagonalMatrix.mat_data[i * size + j] = diag[i];
            }
            else
            {
                diagonalMatrix.mat_data[i * size + j] = 0.0;
            }
        }
    }
    return diagonalMatrix;
}

bool Matrix::is_triangular() const
{
    if (r != c)
    {
        return false;
    }
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            if (mat_data[i * c + j] != 0.0)
            {
                return false;
            }
        }
    }
    return true;
}

void Matrix::setDiagonal(const vector<double> &diagValues)
{
    if (r != c)
    {
        throw invalid_argument("Matrix is not square");
    }
    if (diagValues.size() != r)
    {
        throw invalid_argument("Number of diagonal values does not match matrix size");
    }
    for (size_t i = 0; i < r; ++i)
    {
        mat_data[i * c + i] = diagValues[i];
    }
}

int main()
{
    // Create matrices using various constructors
    Matrix mat1(3, 3, true);                         // Create a 3x3 matrix with random values (uniform distribution)
    Matrix mat2(3, 3, false, 0.0, 1.0);              // Create a 3x3 matrix with random values (normal distribution)
    Matrix mat3 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}; // Create a matrix from an initializer list
    Matrix mat4("example.csv");                      // Load a matrix from a CSV file

    // Print the matrices
    std::cout << "Matrix 1:" << std::endl;
    mat1.print("Additional String for Matrix 1:", mat1.sum());
    std::cout << std::endl;

    std::cout << "Matrix 2:" << std::endl;
    mat2.print("Additional String for Matrix 2:", mat2.sum());
    std::cout << std::endl;

    std::cout << "Matrix 3:" << std::endl;
    mat3.print("Additional String for Matrix 3:", mat3.sum());
    std::cout << std::endl;

    std::cout << "Matrix 4:" << std::endl;
    mat4.print("Additional String for Matrix 4:", mat4.sum());
    std::cout << std::endl;

    // Perform matrix operations
    Matrix result1 = mat1 + mat2;
    std::cout << "Matrix 1 + Matrix 2:" << std::endl;
    result1.print("Additional String for Result 1:", result1.sum());
    std::cout << std::endl;

    Matrix result2 = mat3 * mat4;
    std::cout << "Matrix 3 * Matrix 4:" << std::endl;
    result2.print("Additional String for Result 2:", result2.sum());
    std::cout << std::endl;

    // Other matrix operations
    Matrix mat5(4, 4);
    mat5.setDiagonal({1, 2, 3, 4});
    std::cout << "Matrix 5 (diagonal):" << std::endl;
    mat5.print("Additional String for Matrix 5:", mat5.sum());
    std::cout << std::endl;

    Matrix mat6 = Matrix::diag_matrix({3, 2, 1});
    std::cout << "Matrix 6 (diagonal):" << std::endl;
    mat6.print("Additional String for Matrix 6:", mat6.sum());
    std::cout << std::endl;

    std::cout << "Matrix 3 is triangular: " << (mat3.is_triangular() ? "Yes" : "No") << std::endl;
    std::cout << "Matrix 5 is triangular: " << (mat5.is_triangular() ? "Yes" : "No") << std::endl;

    return 0;
}
