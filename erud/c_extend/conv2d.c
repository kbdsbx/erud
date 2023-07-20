#include <iostream>
#include <Dense>


/**
 * 计算四维张量到一维数组的下标映射
 * si, mi, ni, ci 对应维度的下标
 * s, m, n, c 对应维度的宽度
*/
int index(int si, int mi, int ni, int ci, int s, int m, int n, int c) {
    return si * m * n * c + mi * n * c + ni * c + ci;
}

int index(int pi, int qi, int ci, int p, int q, int c) {
    return pi * q * c + qi * c + ci;
}

/**
 * 卷积前向传播
 * 
 * x : 输入
 * w : 卷积核
 * z : 输出
 * cx : 输出变形后的x的缓存
 * (s, m1, n1, c1) 输入维度
 * (s, m2, n2, c2) 输出维度
 * (p, q, c1, c2) 卷积维度
*/
int conv2d_fprop(
    float* x,
    float* w,
    float* z,
    float* cx,
    int s,
    int m1,
    int n1,
    int c1,
    int m2,
    int n2,
    int p,
    int q,
    int c2,
    int stride
) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _x(p * q * c1, s * m2 * n2);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _w = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> >(w, c2, p * q * c1);
    int _mi = 0, _ni = 0;
    // 将数组元素赋值给变形矩阵
    for (int si = 0; si < s; si++) {
        for (int m2i = 0; m2i < m2; m2i++) {
            for (int n2i = 0; n2i < n2; n2i++) {
                for (int pi = 0; pi < p; pi++) {
                    for (int qi = 0; qi < q; qi++) {
                        for (int c1i = 0; c1i < c1; c1i++) {
                            _mi = stride * m2i + pi;
                            _ni = stride * n2i + qi;
                            _x(index(pi, qi, c1i, p, q, c1), index(si, m2i, n2i, s, m2, n2)) = x[index(si, _mi, _ni, c1i, s, m1, n1, c1)];
                        }
                    }
                }
            }
        }
    }
    // 通过矩阵乘法计算卷积
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _z(_w * _x);

    // 复制到输出
    float* _d;
    _d = _z.data();
    for (int i = 0; i < _z.size(); i++) {
        z[i] = _d[i];
    }

    _d = _x.data();
    for (int i = 0; i < _x.size(); i++) {
        cx[i] = _d[i];
    }

    return 1;
}

/**
 * 卷积反向传播
 * 
 * cx 经过处理的输入的缓存
 * cw 经过处理的输出的缓存
 * dz 导数矩阵的flatten版本
 * dx 输出x梯度
 * dw 输出w梯度
*/
int conv2d_bprop(float* cx, float* cw, float* dz, float* dx, float* dw) {

    return 1;
}

int main() {
    // 2 * 4 * 6 * 3矩阵的flatten版本
    float x[144] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144};
    // 2 * 2 * 3 * 2矩阵的flatten版本
    float w[24] = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24};
    // 计算得2 * 3 * 5 * 2矩阵
    // 步长stride = 1
    float z[60];
    // 缓存大小：s * m2 * n2 * p * q * c1
    float cache_x[2 * 3 * 5 * 2 * 2 * 3];
    float dz[60] = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
       0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,
       0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32,
       0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43,
       0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54,
       0.55, 0.56, 0.57, 0.58, 0.59, 0.6};

    conv2d_fprop(x, w, z, cache_x, 2, 4, 6, 3, 3, 5, 2, 2, 2, 1);
    std::cout << "Z = ( ";
    for (int i = 0; i < 2*3*5*2; i++) {
        std::cout << z[i] << " ";
    }
    std::cout << ")" << std::endl;
}