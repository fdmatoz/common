import numpy as np 
import gpumd as md


particles = []

Num=50
l0 = 1.0
for i in range(0, Num):
    pos = md.real3(1.0 + i*l0, 0.0, 0.0)
    particles.append(pos)
    #print(pos)


rattle = md.rattle_constraint()
rattle.l0 = 2**(1/6)
rattle.tol = 1e-9
#print('Before',particles)
#print('After',rattle.enforce(particles))
particles_new=rattle.enforce(particles)
particles_np = np.array([[p.x, p.y, p.z] for p in particles_new])
for i in range(0,Num):
    print(i,particles_np[i])
    
    