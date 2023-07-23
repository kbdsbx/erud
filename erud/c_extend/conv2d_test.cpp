#include "conv2d.cpp"
#include <iostream>
#include <vector>
#include <tuple>


int main() {
    // 2 * 4 * 6 * 3矩阵的flatten版本
    // 144
    std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144};
    // 2 * 2 * 3 * 2矩阵的flatten版本
    // 24
    std::vector<float> w = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24};
    // 计算得2 * 3 * 5 * 2矩阵
    // 步长stride = 1
    // 60
    std::vector<float> z(60);
    // 缓存大小：s * m2 * n2 * p * q * c1
    // 2 * 3 * 5 * 2 * 2 * 3 = 360
    std::vector<float> cache_x(360);
    // 60
    std::vector<float> dz = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
       0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,
       0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32,
       0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43,
       0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54,
       0.55, 0.56, 0.57, 0.58, 0.59, 0.6};

    const auto v = conv2d_fprop(x, w, 2, 4, 6, 3, 3, 5, 2, 2, 2, 1);
    z = std::get<0>(v);
    cache_x = std::get<1>(v);

    // 2 * 2 * 3 * 2
    // 24
    std::vector<float> dw(24);
    // 2 * 4 * 6 * 3
    // 144
    std::vector<float> dx(144);

    conv2d_bprop(cache_x, w, dz, dx, dw, 2, 4, 6, 3, 3, 5, 2, 2, 2, 1);

    std::cout << "Z = ( ";
    for (int i = 0; i < 2*3*5*2; i++) {
        std::cout << z[i] << " ";
    }
    std::cout << ")" << std::endl;

    std::cout << "dX = ( ";
    for (int i = 0; i < 144; i++) {
        std::cout << dx[i] << " ";
    }
    std::cout << ")" << std::endl;

    std::cout << "dW = ( ";
    for (int i = 0; i < 24; i++) {
        std::cout << dw[i] << " ";
    }
    std::cout << ")" << std::endl;

    return 0;
}