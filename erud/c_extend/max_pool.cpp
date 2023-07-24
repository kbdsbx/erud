#include "Eigen/Dense"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/**
 * 计算四维张量到一维数组的下标映射
 * si, mi, ni, ci 对应维度的下标
 * s, m, n, c 对应维度的宽度
*/
inline int index(int si, int mi, int ni, int ci, int s, int m, int n, int c) {
    return si * m * n * c + mi * n * c + ni * c + ci;
}


/**
 * 最大池化
 *
 * x 输入
 * z 输出
 * cx 与输入相同维度的全0缓存，用来记录最大值位置
 * (s, m1, n1, c) 输入维度
 * (s, m2, n2, c) 输出维度
 * (p, q) 池化维度
 * stride 池化步长
*/
int max_pool_fprop(
    // float* x,
    // float* z,
    // int* cx,
    py::array_t<float>& x,
    py::array_t<float>& z,
    py::array_t<int>& cx,
    int s,
    int m1,
    int n1,
    int m2,
    int n2,
    int p,
    int q,
    int c,
    int stride
) {
    py::buffer_info xb = x.request();
    float* xp = (float*)xb.ptr;
    auto cxs = cx.mutable_unchecked<1>();
    auto zs = z.mutable_unchecked<1>();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _w(p, q);
    int maxR, maxC;
    int _mi = 0, _ni = 0;
    for (int si = 0; si < s; si++) {
        for (int m2i = 0; m2i < m2; m2i++) {
            for (int n2i = 0; n2i < n2; n2i++) {
                for (int ci = 0; ci < c; ci++) {
                    for (int pi = 0; pi < p; pi++) {
                        for (int qi = 0; qi < q; qi++) {
                            _mi = stride * m2i + pi;
                            _ni = stride * n2i + qi;
                            _w(pi, qi) = xp[index(si, _mi, _ni, ci, s, m1, n1, c)];
                        }
                    }
                    zs(index(si, m2i, n2i, ci, s, m2, n2, c)) = _w.maxCoeff(&maxR, &maxC);
                    // 记录池化中最大值的位置，为了反向传播
                    cxs(index(si, stride * m2i + maxR, stride * n2i + maxC, ci, si, m1, n1, c)) = 1;
                }
            }
        }
    }

    return 0;
}


/**
 * 最大池化反向传播
 *
 * cx 缓存，x中最大值的位置矩阵
 * dz 导数矩阵
 * dx 输出
 * (s, m1, n1, c) 输入维度
 * (s, m2, n2, c) 输出维度
 * (p, q) 池化维度
 * stride 池化步长
*/
int max_pool_bprop(
    // int* cx,
    // float* dz,
    // float* dx,
    py::array_t<int>& cx,
    py::array_t<float>& dz,
    py::array_t<float>& dx,
    int s,
    int m1,
    int n1,
    int m2,
    int n2,
    int p,
    int q,
    int c,
    int stride
) {
    py::buffer_info cxb = cx.request();
    py::buffer_info dzb = dz.request();
    int* cxp = (int*)cxb.ptr;
    float* dzp = (float*)dzb.ptr;
    
    auto dxs = dx.mutable_unchecked<1>();

    int _mi, _ni, _idx;
    for (int si = 0; si < s; si++) {
        for (int m2i = 0; m2i < m2; m2i++) {
            for (int n2i = 0; n2i < n2; n2i++) {
                for (int ci = 0; ci < c; ci++) {
                    for (int pi = 0; pi < p; pi++) {
                        for (int qi = 0; qi < q; qi++) {
                            _mi = stride * m2i + pi;
                            _ni = stride * n2i + qi;
                            _idx = index(si, _mi, _ni, ci, s, m1, n1, c);
                            dxs(_idx) = cxp[_idx] * dzp[index(si, m2i, n2i, ci, s, m2, n2, c)];
                        }
                    }
                }
            }
        }
    }

    return 0;
}


PYBIND11_MODULE(max_pool, m) {
    m.doc() = "C++ extends in key computation.";
    m.def(
        "max_pool_fprop",
        &max_pool_fprop,
        py::arg(),
        py::arg{}.noconvert(),
        py::arg{}.noconvert(),
        py::arg(),
        py::arg(),
        py::arg(),
        py::arg(),
        py::arg(),
        py::arg(),
        py::arg(),
        py::arg(),
        py::arg()
    );
    m.def(
        "max_pool_bprop",
        &max_pool_bprop,
        py::arg(),
        py::arg(),
        py::arg{}.noconvert(),
        py::arg(),
        py::arg(),
        py::arg(),
        py::arg(),
        py::arg(),
        py::arg(),
        py::arg(),
        py::arg(),
        py::arg()
    );
}
