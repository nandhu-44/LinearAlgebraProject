#include "Matrix.h"
#include "nonMemberFunctions.cpp"

int main() {
    // Create matrices using various constructors
    Matrix mat1(3, 3, true); // Create a 3x3 matrix with random values (uniform distribution)
    Matrix mat2(3, 3, false, 0.0, 1.0); // Create a 3x3 matrix with random values (normal distribution)
    Matrix mat3 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}; // Create a matrix from an initializer list
    Matrix mat4("example.csv"); // Load a matrix from a CSV file

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

    Matrix result1 = mat1 + mat2;
    std::cout << "Matrix 1 + Matrix 2:" << std::endl;
    result1.print("Additional String for Result 1:", result1.sum());
    std::cout << std::endl;

    Matrix result2 = mat3 * mat4;
    std::cout << "Matrix 3 * Matrix 4:" << std::endl;
    result2.print("Additional String for Result 2:", result2.sum());
    std::cout << std::endl;

    // Other matrix operations
    std::vector<double> column = mat3.column(1);
    std::cout << "Column 1 of Matrix 3:" << std::endl;
    for (double value : column) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    Matrix submat = mat4.sub_matrix(0, 0, 1, 1);
    std::cout << "Submatrix of Matrix 4:" << std::endl;
    submat.print("Additional String for Submatrix:", submat.sum());
    std::cout << std::endl;


    return 0;
}