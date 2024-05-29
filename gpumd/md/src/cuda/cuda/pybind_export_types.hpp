/************************************************************************************
* MIT License                                                                       *
*                                                                                   *
* Copyright (c) 2020 Dr. Daniel Alejandro Matoz Fernandez                           *
*               fdamatoz@gmail.com                                                  *
* Permission is hereby granted, free of charge, to any person obtaining a copy      *
* of this software and associated documentation files (the "Software"), to deal     *
* in the Software without restriction, including without limitation the rights      *
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         *
* copies of the Software, and to permit persons to whom the Software is             *
* furnished to do so, subject to the following conditions:                          *
*                                                                                   *
* The above copyright notice and this permission notice shall be included in all    *
* copies or substantial portions of the Software.                                   *
*                                                                                   *
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        *
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          *
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       *
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            *
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     *
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     *
* SOFTWARE.                                                                         *
*************************************************************************************/
#ifndef __pybind_export_types_hpp__
#define __pybind_export_types_hpp__

#include "globaltypes.hpp"
#include "box.hpp"

/*  @Note In the same way that we define operators in c++ 
    (see system/particleoperators.hpp), in python is possible to define what is call
    magic methods https://rszalski.github.io/magicmethods/
    this methods are used to perfom operations in classes
    a trival example is given here for the real3 class
    (see types/pybind_export_types.hpp)
*/
void export_real3(py::module &m)
{
    py::class_<real3>(m, "real3")
        .def(py::init<>())
        .def("__init__", [](real3 &instance, real x, real y, real z) {
            new (&instance) real3();
            instance.x = x;
            instance.y = y;
            instance.z = z;
        })
        .def("__repr__", [](const real3 &a) {
            return ("<real3 x = " + std::to_string(a.x) + " y = " + std::to_string(a.y) + " z = " + std::to_string(a.z) + " >");
        })
        .def_readwrite("x", &real3::x)
        .def_readwrite("y", &real3::y)
        .def_readwrite("z", &real3::z)

        /*.def("minimum_image", [](const real3 &a, const real3 &b, const BoxType &box) 
        {
            return(host::vector_subtract(a,b,box));          
        })
        .def("need_wrapping", [](const real3 &a, const real3 &b, const BoxType &box) 
        {
            return(host::need_wrapping(a,b,box));          
        })*/
        /*opeators*/
        .def(
            "__mul__", [](const real3 &a, const real3 &b) {
                return (vdot(a, b));
            },
            py::is_operator())
        .def(
            "__abs__", [](const real3 &a) {
                return (sqrt(vdot(a, a)));
            },
            py::is_operator())
        .def(
            "__add__", [](const real3 &a, const real3 &b) {
                real3 c;
                vsum(c, a, b);
                return (c);
            },
            py::is_operator())
        .def(
            "__sub__", [](const real3 &a, const real3 &b) {
                real3 c;
                vsub(c, a, b);
                return (c);
            },
            py::is_operator())
        .def(
            "__matmul__", [](const real3 &a, const real3 &b) {
                real3 c;
                vcross(c, a, b);
                return (c);
            },
            py::is_operator())
        .def(
            "__neg__", [](const real3 &a) {
                real3 c;
                c.x = -a.x;
                c.y = -a.y;
                c.z = -a.z;
                return (c);
            },
            py::is_operator())
        .def(
            "__pow__", [](const real3 &a, const real &b) {
                real3 c;
                c.x = pow(a.x, b);
                c.y = pow(a.y, b);
                c.z = pow(a.z, b);
                return (c);
            },
            py::is_operator())
        .def(
            "__iadd__", [](const real3 &a, const real3 &b) {
                real3 c;
                vsum(c, a, b);
                return (c);
            },
            py::is_operator())
        .def(
            "__isub__", [](const real3 &a, const real3 &b) {
                real3 c;
                vsub(c, a, b);
                return (c);
            },
            py::is_operator())
        .def(
            "__imul__", [](const real3 &a, const real3 &b) {
                return (vdot(a, b));
            },
            py::is_operator())
        .def(
            "__scale__", [](const real3 &a, const real &b) {
                real3 c = a;
                c.x *= b;
                c.y *= b;
                c.z *= b;
                return c;
            },
            py::is_operator());
}

void export_inth3(py::module &m)
{
    py::class_<inth3>(m, "inth3")
        .def(py::init<>())
        .def("__init__", [](inth3 &instance, int x, int y, int z) {
            new (&instance) inth3();
            instance.x = x;
            instance.y = y;
            instance.z = z;
        })
        .def("__repr__", [](const inth3 &a) {
            return ("<inth3 x = " + std::to_string(a.x) + " y = " + std::to_string(a.y) + " z = " + std::to_string(a.z) + " >");
        })
        .def_readwrite("x", &inth3::x)
        .def_readwrite("y", &inth3::y)
        .def_readwrite("z", &inth3::z)
        /*opeators*/
        .def(
            "__mul__", [](const inth3 &a, const inth3 &b) {
                return (vdot(a, b));
            },
            py::is_operator())
        .def(
            "__abs__", [](const inth3 &a) {
                return (sqrt(vdot(a, a)));
            },
            py::is_operator())
        .def(
            "__add__", [](const inth3 &a, const inth3 &b) {
                inth3 c;
                vsum(c, a, b);
                return (c);
            },
            py::is_operator())
        .def(
            "__sub__", [](const inth3 &a, const inth3 &b) {
                inth3 c;
                vsub(c, a, b);
                return (c);
            },
            py::is_operator())
        .def(
            "__matmul__", [](const inth3 &a, const inth3 &b) {
                inth3 c;
                vcross(c, a, b);
                return (c);
            },
            py::is_operator())
        .def(
            "__neg__", [](const inth3 &a) {
                inth3 c;
                c.x = -a.x;
                c.y = -a.y;
                c.z = -a.z;
                return (c);
            },
            py::is_operator())
        .def(
            "__pow__", [](const inth3 &a, const real &b) {
                inth3 c;
                c.x = pow(a.x, b);
                c.y = pow(a.y, b);
                c.z = pow(a.z, b);
                return (c);
            },
            py::is_operator())
        .def(
            "__iadd__", [](const inth3 &a, const inth3 &b) {
                inth3 c;
                vsum(c, a, b);
                return (c);
            },
            py::is_operator())
        .def(
            "__isub__", [](const inth3 &a, const inth3 &b) {
                inth3 c;
                vsub(c, a, b);
                return (c);
            },
            py::is_operator())
        .def(
            "__imul__", [](const inth3 &a, const inth3 &b) {
                return (vdot(a, b));
            },
            py::is_operator())
        .def(
            "__scale__", [](const inth3 &a, const int &b) {
                inth3 c = a;
                c.x *= b;
                c.y *= b;
                c.z *= b;
                return c;
            },
            py::is_operator());
}

void export_bool3(py::module &m)
{
    py::class_<bool3>(m, "bool3")
        .def(py::init<>())
        .def("__init__", [](bool3 &instance, bool x, bool y, bool z) {
            new (&instance) bool3();
            instance.x = x;
            instance.y = y;
            instance.z = z;
        })
        .def("__repr__", [](const bool3 &a) {
            return ("<bool3 x = " + std::to_string(a.x) + " y = " + std::to_string(a.y) + " z = " + std::to_string(a.z) + " >");
        })
        .def_readwrite("x", &bool3::x)
        .def_readwrite("y", &bool3::y)
        .def_readwrite("z", &bool3::z);
}
#endif
