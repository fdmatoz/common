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
#ifndef __typesmath_hpp__
#define __typesmath_hpp__

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "globaltypes.hpp"
#include "../configuration/dev_types.hpp"

//! make a real2 value
DEV_CALLABLE_INLINE real2 make_real2(real x, real y)
{
    real2 val;
    val.x = x;
    val.y = y;
    return val;
}

//! make a real3 value
DEV_CALLABLE_INLINE real3 make_real3(real x, real y, real z)
{
    real3 val;
    val.x = x;
    val.y = y;
    val.z = z;
    return val;
}

//! make a real4 value
DEV_CALLABLE_INLINE real4 make_real4(real x, real y, real z, real w)
{
    real4 val;
    val.x = x;
    val.y = y;
    val.z = z;
    val.w = w;
    return val;
}

namespace fastmath
{
    //! compute the square root of a real
    namespace device
    {
        //! Use CUDA intrinsics for sqrt
        DEV_INLINE_LAMBDA float sqrt(float x)
        {
            return ::__fsqrt_rd(x);
        }
        DEV_INLINE_LAMBDA double sqrt(double x)
        {
            return ::sqrt(x);
        }
    } // namespace device
    DEV_CALLABLE_INLINE float sqrt(float x)
    {
        return ::sqrtf(x);
    }
    DEV_CALLABLE_INLINE double sqrt(double x)
    {
        return ::sqrt(x);
    }
    //! compute the inverse square root of a real
    namespace device
    {
        //! Use CUDA intrinsics for sqrt
        DEV_INLINE_LAMBDA float rsqrt(float x)
        {
            return ::__frsqrt_rn(x);
        }
        DEV_INLINE_LAMBDA double rsqrt(double x)
        {
            return 1.0f / ::sqrt(x);
        }
    } // namespace device
    DEV_CALLABLE_INLINE float rsqrt(float x)
    {
        return 1.0f / ::sqrtf(x);
    }
    DEV_CALLABLE_INLINE double rsqrt(double x)
    {
        return 1.0f / ::sqrt(x);
    }
    //! compute the cosine a real
    namespace device
    {
        //! Use CUDA intrinsics for sqrt
        DEV_INLINE_LAMBDA float cos(float x)
        {
            return ::__cosf(x);
        }
        DEV_INLINE_LAMBDA double cos(double x)
        {
            return cos(x);
        }
    } // namespace device
    DEV_CALLABLE_INLINE float cos(float x)
    {
        return cosf(x);
    }
    DEV_CALLABLE_INLINE double cos(double x)
    {
        return cos(x);
    }
    //! compute the acos a real
    namespace device
    {
        //! Use CUDA intrinsics for sqrt
        DEV_INLINE_LAMBDA float acos(float x)
        {
            return ::acosf(x);
        }
        DEV_INLINE_LAMBDA double acos(double x)
        {
            return acos(x);
        }
    } // namespace device
    DEV_CALLABLE_INLINE float acos(float x)
    {
        return acosf(x);
    }
    DEV_CALLABLE_INLINE double acos(double x)
    {
        return acos(x);
    }    
    //! compute the sine a real
    namespace device
    {
        //! Use CUDA intrinsics for sqrt
        DEV_INLINE_LAMBDA float sin(float x)
        {
            return ::__sinf(x);
        }
        DEV_INLINE_LAMBDA double sin(double x)
        {
            return sin(x);
        }
    } // namespace device
    DEV_CALLABLE_INLINE float sin(float x)
    {
        return sinf(x);
    }
    DEV_CALLABLE_INLINE double sin(double x)
    {
        return sin(x);
    }
    //! compute the cosine and sin of a real simultaneously
    namespace device
    {
        //! Use CUDA intrinsics for sqrt
        DEV_INLINE_LAMBDA void sincos(float x, float &sx, float &cx)
        {
            __sincosf(x, &sx, &cx);
        }
        DEV_INLINE_LAMBDA void sincos(double x, double &sx, double &cx)
        {
            sx = sin(x);
            cx = cos(x);
        }
    } // namespace device
    DEV_CALLABLE_INLINE void sincos(float x, float &sx, float &cx)
    {
        sincosf(x, &sx, &cx);
    }
    DEV_CALLABLE_INLINE void sincos(double x, double &sx, double &cx)
    {
        sx = sin(x);
        cx = cos(x);
    }
    namespace device
    {
        //! Use CUDA intrinsics for exp
        DEV_INLINE_LAMBDA float exp(float x)
        {
            return ::expf(x);
        }
        DEV_INLINE_LAMBDA double tan(double x)
        {
            return exp(x);
        }
    } // namespace device
      //! Use CUDA intrinsics for exp
    DEV_CALLABLE_INLINE float exp(float x)
    {
        return ::expf(x);
    }
    DEV_CALLABLE_INLINE double tan(double x)
    {
        return exp(x);
    }

    //! compute the absolute value of a real
    namespace device
    {
        //! Use CUDA intrinsics for sqrt
        DEV_INLINE_LAMBDA float fabs(float x)
        {
            return ::fabsf(x);
        }
        DEV_INLINE_LAMBDA double fabs(double x)
        {
            return fabs(x);
        }
    } // namespace device
      //! Use CUDA intrinsics for sqrt
    DEV_CALLABLE_INLINE float fabs(float x)
    {
        return ::fabsf(x);
    }
    DEV_CALLABLE_INLINE double fabs(double x)
    {
        return fabs(x);
    }
    //! compute the atan2
    namespace device
    {
        //! Use CUDA intrinsics for sqrt
        DEV_INLINE_LAMBDA float atan2(float y, float x)
        {
            return ::atan2f(y, x);
        }
        DEV_INLINE_LAMBDA double atan2(double y, double x)
        {
            return atan2(y, x);
        }
    } // namespace device
      //! Use CUDA intrinsics for sqrt
    DEV_CALLABLE_INLINE float atan2(float y, float x)
    {
        return ::atan2f(y, x);
    }
    DEV_CALLABLE_INLINE double atan2(double y, double x)
    {
        return atan2(y, x);
    }
    /******************************************************************************/
    /// Vector Math
    /******************************************************************************/
    //! compute the dot product of two real3 vectors
    namespace device
    {
        //! Use CUDA intrinsics
        DEV_INLINE_LAMBDA float dot_real3(const float3& a, const float3& b)
        {
            // Compute as  axb + z a single operation, in round-to-nearest-even mode.
            float result = __fmaf_rn(a.x, b.x, __fmaf_rn(a.y, b.y, a.z * b.z));
            return result;
        }
        DEV_INLINE_LAMBDA double dot_real3(const double3& a, const double3& b)
        {
            double result = a.x * b.x + a.y * b.y + a.z * b.z;
            return result;
        }
        //! Use CUDA intrinsics
        DEV_INLINE_LAMBDA real3 cross_real3(const float3& a, const float3& b)
        {
            // Compute as  axb + z a single operation, in round-to-nearest-even mode.
            //float v_x = __fmaf_rn(a.y, b.z, -a.z * b.y);
            //float v_y = __fmaf_rn(a.z, b.x, -a.x * b.z);
            //float v_z = __fmaf_rn(a.x, b.y, -a.y * b.x);
            // v.x = v1.y * v2.z - v1.z * v2.y)
            // v.y = v1.z * v2.x - v1.x * v2.z)
            // v.z = v1.x * v2.y - v1.y * v2.x)
            return make_real3(__fmaf_rn(a.y, b.z, -a.z * b.y), 
                              __fmaf_rn(a.z, b.x, -a.x * b.z), 
                              __fmaf_rn(a.x, b.y, -a.y * b.x));
        
        }
        DEV_INLINE_LAMBDA real3 cross_real3(const double3& a, const double3& b)
        {
            real3 c;
            vcross(c, a, b);
            return c;
        }
    } // namespace device
} // namespace fastmath /**/

#endif
