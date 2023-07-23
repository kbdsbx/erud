#include "Eigen/Dense"
#include <vector>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <stdio.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

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

// PYBIND11_MAKE_OPAQUE(std::vector<float>);


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
 * stride 卷积步长
*/
// std::tuple<std::vector<float>, std::vector<float> >  
int conv2d_fprop(
    py::array_t<float>& x,
    py::array_t<float>& w,
    py::array_t<float>& z,
    py::array_t<float>& cx,
    int s,
    int m1,
    int n1,
    int c1,
    int m2,
    int n2,
    int c2,
    int p,
    int q,
    int stride
) {
    py::buffer_info xb = x.request();
    py::buffer_info wb = w.request();
    float* xp = (float*)xb.ptr;
    float* wp = (float*)wb.ptr;
    // std::vector<float> z;
    // std::vector<float> cx;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _x(p * q * c1, s * m2 * n2);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _w = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> >(wp, c2, p * q * c1);
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
                            _x(index(pi, qi, c1i, p, q, c1), index(si, m2i, n2i, s, m2, n2)) = xp[index(si, _mi, _ni, c1i, s, m1, n1, c1)];
                        }
                    }
                }
            }
        }
    }
    // 通过矩阵乘法计算卷积
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _z(_w * _x);

    auto cxs = cx.mutable_unchecked<1>();
    auto zs = z.mutable_unchecked<1>();

    // 复制到输出
    float* _d;
    _d = _z.data();
    for (int i = 0; i < _z.size(); i++) {
        zs(i) = _d[i];
    }

    _d = _x.data();
    for (int i = 0; i < _x.size(); i++) {
        cxs(i) = _d[i];
    }
    return 1;
}

/**
 * 卷积反向传播
 *
 * cx 经过处理的输入的缓存
 * w 卷积核
 * dz 导数矩阵的flatten版本
 * dx 输出x梯度
 * dw 输出w梯度
 * (s, m1, n1, c1) 输入维度
 * (s, m2, n2, c2) 输出维度
 * (p, q, c1, c2) 卷积维度
 * stride 卷积步长
*/
int conv2d_bprop(
    py::array_t<float>& cx,
    py::array_t<float>& w,
    py::array_t<float>& dz,

    py::array_t<float>& dx,
    py::array_t<float>& dw,
    int s,
    int m1,
    int n1,
    int c1,
    int m2,
    int n2,
    int c2,
    int p,
    int q,
    int stride
) {
    py::buffer_info cxb = cx.request();
    py::buffer_info wb = w.request();
    py::buffer_info dzb = dz.request();
    float* cxp = (float*)cxb.ptr;
    float* wp = (float*)wb.ptr;
    float* dzp = (float*)dzb.ptr;

    auto dxs = dx.mutable_unchecked<1>();
    auto dws = dw.mutable_unchecked<1>();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _w = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> >(wp, c2, p * q * c1);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _cx = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> >(cxp, p * q * c1, s * m2 * n2);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _dz = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> >(dzp, c2, s * m2 * n2);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _wt(_w.transpose());
    // (p * q * c1, s * m2 * n2)
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _dtx(_wt * _dz);
    int _mi = 0, _ni = 0;
    for (int si = 0; si < s; si++) {
        for (int m2i = 0; m2i < m2; m2i++) {
            for (int n2i = 0; n2i < n2; n2i++) {
                for (int pi = 0; pi < p; pi++) {
                    for (int qi = 0; qi < q; qi++) {
                        for (int c1i = 0; c1i < c1; c1i++) {
                            _mi = stride * m2i + pi;
                            _ni = stride * n2i + qi;
                            dxs(index(si, _mi, _ni, c1i, s, m1, n1, c1)) += _dtx(index(pi, qi, c1i, p, q, c1), index(si, m2i, n2i, s, m2, n2));
                        }
                    }
                }
            }
        }
    }

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _dw(_dz * _cx.transpose());

    // dw复制到输出
    float* _d;
    _d = _dw.data();
    for (int i = 0; i < _dw.size(); i++) {
        dws(i) = _d[i];
    }

    return 1;
}

PYBIND11_MODULE(conv2d, m) {
    m.doc() = "C++ extends in key computation.";
    m.def(
        "conv2d_fprop", 
        &conv2d_fprop, 
        py::arg(), 
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
        py::arg(), 
        py::arg()
    );
    m.def(
        "conv2d_bprop", 
        &conv2d_bprop, 
        py::arg(), 
        py::arg(), 
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
        py::arg(), 
        py::arg()
    );
}
