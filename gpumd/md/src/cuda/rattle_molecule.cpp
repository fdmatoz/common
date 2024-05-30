#include "cuda_runtime.h"
#include <thrust/find.h>

#include "rattle_molecule.hpp"
#include "computevector.hpp"
#include "atomic_add_double.hpp"
#include "atomic_exch_double.hpp"
__global__
void rattle_kernel( const int Numparticles,
                    real3* particles,
                    const real3* particles_old,
                    int* move,
                    const int* moved,
                    const real sq_l0,
                    const real sq_tol)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < Numparticles-1;
         i += blockDim.x * gridDim.x)
         
    {
        
        auto j = i + 1;  //Partner atom in this constraint
        //printf("particle_ij[%i,%i]\n", i,j);
        
        if(moved[i] == 1 || moved[j] == 1)
        { 
            
            auto rij = device::vector_subtract(particles[i], particles[j]);
            
            auto diffsq = sq_l0 - device::sq_norm(rij);
            
            if(fabs(diffsq) > sq_tol*sq_l0) // Test whether constraint not already satisfied
            {
                
                auto rij_old = device::vector_subtract(particles_old[i], particles_old[j]);
                auto dot = vdot(rij_old, rij); // This should be of the order of bond**2
                auto g = diffsq / ( 4.0 * dot );
                auto dr = device::vector_constant_mul(rij_old, g); 

                device::double_atomicAdd(&(particles[i].x), dr.x);
                device::double_atomicAdd(&(particles[i].y), dr.y);
                device::double_atomicAdd(&(particles[i].z), dr.z);
                
                /*
                From the tests I have done if you have many particles (>20) with this some particles will have the same positions
                 
                device::double_atomicAdd(&(particles[j].x), -dr.x);
                device::double_atomicAdd(&(particles[j].y), -dr.y);
                device::double_atomicAdd(&(particles[j].z), -dr.z);
               */
            
                auto move_i = atomicExch(&move[i], 1); // I think this is nort working because the for is looping till max_iter (it's this part or d_moved_iter = thrust::find(d_moved.begin(), d_moved.end(), 1);)
                auto move_j = atomicExch(&move[j], 1);
               
                //printf("move[%i,%i]\n", move_i,move_j);

            }
        }
        //printf("particles[%i] -> pos = (%f, %f, %f)\n", i, particles[i].x, particles[i].y, particles[i].z);
    }
}


host::vector<real3> RattleMoleculeClass::enforce_constraint(const host::vector<real3>& particles)
{
    //define an empty array (dynamical) in gpu
    device::vector<real3> d_particles;
    //Copy the host vector into the gpu
    d_particles = particles;
    std::cout << "Particles SIze " << d_particles.size() << std::endl;


    device::vector<int> d_moved (d_particles.size(), 1);
    device::vector<int> d_move (d_particles.size(), 0);

    /*
    
    In here your GPU code goes
    */
    // (gpu<->gpu copy)
    device::vector<real3> d_particles_old = d_particles;
    int k=0;
    for(auto iter = 0; iter<this->m_max_iter; iter++)
    {
        rattle_kernel<<<256, 512>>>(d_particles.size(),
                                    device::raw_pointer_cast(&d_particles[0]),
                                    device::raw_pointer_cast(&d_particles_old[0]),
                                    device::raw_pointer_cast(&d_move[0]),
                                    device::raw_pointer_cast(&d_moved[0]),
                                    this->m_l0*this->m_l0,
                                    this->m_tol*this->m_tol
                                    );
        k=k+1;
        printf("iteration[%i]\n", k);


        //Copy move to moved (gpu<->gpu copy)
        d_moved = d_move;
        //d_particles_old = d_particles;

        //Check if all of the tolerances are satisfied
        
        device::vector<int>::iterator d_moved_iter;
        d_moved_iter = thrust::find(d_moved.begin(), d_moved.end(), 1); 
        
        if(d_moved_iter == d_moved.end())
            break;

    }
    //(gpu->cpu copy)
    return device::copy(d_particles);
}
