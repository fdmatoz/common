#ifndef __rattle_molecule_hpp__
#define __rattle_molecule_hpp__

#include "globaltypes.hpp"
#include "devicevector.hpp"
#include "hostvector.hpp"
class RattleMoleculeClass
{
    public:
    RattleMoleculeClass(): m_tol(1e-6), m_l0(1.0), m_max_iter(1000)
    {}
    ~RattleMoleculeClass(){}

    void set_tolerance(const real& tol)
    {
        m_tol = tol;
    }
    auto get_tolerance()
    {
        return m_tol;
    }

    void set_l0(const real& l0)
    {
        m_l0 = l0;
    }
    auto get_l0()
    {
        return m_l0;
    }


    host::vector<real3> enforce_constraint(const host::vector<real3>& particles);

    private:
    real m_tol;
    real m_l0;
    int m_max_iter;
    
};


#endif