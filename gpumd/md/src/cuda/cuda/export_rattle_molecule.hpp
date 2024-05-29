#ifndef __export_rattle_molecule_hpp__
#define __export_rattle_molecule_hpp__

#include "rattle_molecule.hpp"

void export_rattle_molecule(py::module &m)
{
    py::class_<RattleMoleculeClass>(m, "rattle_constraint")
        .def(py::init<>())
        .def_property("tol", &RattleMoleculeClass::get_tolerance, &RattleMoleculeClass::set_tolerance, "tolerance rattle method")
        .def_property("l0", &RattleMoleculeClass::get_l0, &RattleMoleculeClass::set_l0, "l0 rattle method")

        .def("enforce", &RattleMoleculeClass::enforce_constraint)
        ;
}

#endif