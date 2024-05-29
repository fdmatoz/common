#ifndef __computevector_hpp__
#define __computevector_hpp__

#include "globaltypes.hpp"
#include "../configuration/dev_types.hpp"
#include "../box/pbc_device.hpp"

namespace device
{
    /** @defgroup ComputeGPUfn Geometry
*  @brief ComputeGeometry computes properties in Mesh or the space
*  @{
*/
    DEV_CALLABLE_INLINE
    real3 unit_vector(const real3 &v)
    {
        real norm = sqrt(vdot(v, v));
        real3 v_unit = v;
        v_unit.x /= norm;
        v_unit.y /= norm;
        v_unit.z /= norm;
        return (v_unit);
    }

    DEV_CALLABLE_INLINE
    real3 vector_cross(const real3 &v1, const real3 &v2)
    {
        real3 v;
        vcross(v, v1, v2);
        return v;
    }

    DEV_CALLABLE_INLINE
    real3 vector_sum(const real3 &v1, const real3 &v2)
    {
        real3 v;
        v.x = v1.x + v2.x;
        v.y = v1.y + v2.y;
        v.z = v1.z + v2.z;
        return v;
    }

    DEV_CALLABLE_INLINE
    real3 vector_sum(const real3 &v1, const real3 &v2, const Box &box)
    {
        real3 v;
        v.x = v1.x + v2.x;
        v.y = v1.y + v2.y;
        v.z = v1.z + v2.z;
        device::enforce_periodic(v, Box);
        return v;
    }

    DEV_CALLABLE_INLINE
    real3 vector_subtract(const real3 &v1, const real3 &v2)
    {
        real3 v;
        v.x = v1.x - v2.x;
        v.y = v1.y - v2.y;
        v.z = v1.z - v2.z;
        return v;
    }

    DEV_CALLABLE_INLINE
    real3 vector_subtract(const real3 &v1, const real3 &v2, const Box &box)
    {
        return (device::minImage(v2, v1, box));
    }
    /*! @} */
} // namespace device
#endif
/*! @} */
