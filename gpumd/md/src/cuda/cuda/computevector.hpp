#ifndef __computevector_hpp__
#define __computevector_hpp__

#include "globaltypes.hpp"

namespace device
{
    /** @defgroup ComputeGPUfn Geometry
*  @brief ComputeGeometry computes properties in Mesh or the space
*  @{
*/
    __device__
    real3 unit_vector(const real3 &v)
    {
        real norm = sqrt(vdot(v, v));
        real3 v_unit = v;
        v_unit.x /= norm;
        v_unit.y /= norm;
        v_unit.z /= norm;
        return (v_unit);
    }

    __device__
    real3 vector_cross(const real3 &v1, const real3 &v2)
    {
        real3 v;
        vcross(v, v1, v2);
        return v;
    }

    __device__
    real3 vector_sum(const real3 &v1, const real3 &v2)
    {
        real3 v;
        v.x = v1.x + v2.x;
        v.y = v1.y + v2.y;
        v.z = v1.z + v2.z;
        return v;
    }

    __device__
    real3 vector_subtract(const real3 &v1, const real3 &v2)
    {
        real3 v;
        v.x = v1.x - v2.x;
        v.y = v1.y - v2.y;
        v.z = v1.z - v2.z;
        return v;
    }
    __device__
    real sq_norm(const real3 &v1)
    {
        return vdot(v1, v1);
    }

    __device__
    real3 vector_constant_mul(real3 v1, const real& a)
    {
       v1.x*=a;
       v1.y*=a;
       v1.z*=a;
        return v1;
    }
    /*! @} */
} // namespace device
#endif
/*! @} */
