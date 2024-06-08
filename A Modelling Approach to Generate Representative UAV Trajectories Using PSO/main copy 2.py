'''
Author: sourcedream
Date: 2024-06-05 22:12:01
LastEditTime: 2024-06-06 21:47:29
Description: A Modelling Approach to Generate Representative UAV Trajectories Using PSO
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

obstacles_num = 10  
map_size = np.array([120, 120, 120])  
start = np.array([10, 10, 10])  
target = np.array([100, 100, 80])  
particle_num = 40  
particle_point_num = 2  
iteration_num = 200  
w = 1.2  
c1 = 2  
c2 = 2  
figure = plt.figure()  
scene = figure.add_subplot(111, projection='3d')  
scene.set_title('PSO for UAV Trajectory Generation')  
scene.set_xlabel('X')  
scene.set_ylabel('Y')  
scene.set_zlabel('Z')  
scene.scatter(*start, color='green', label='Start')  
scene.scatter(*target, color='red', label='Target')  
obstacles = []  
for i in range(obstacles_num):
    center = np.random.rand(3) * map_size
    radius = np.random.rand() * 30
    obstacles.append((center, radius))
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    scene.plot_surface(x, y, z, color='gray')
def check_collision(path, obstacles):
    for point in path:
        for center, radius in obstacles:
            if np.linalg.norm(point - center) < radius:
                return True
    return False
particles = np.random.rand(particle_num, particle_point_num, 3) * map_size  
particles_velocity = np.zeros((particle_num, particle_point_num, 3))  
particles_best = particles.copy()  
particles_best_fitness = np.full(particle_num, np.inf)  
global_best = particles[0].copy()  
global_best_fitness = np.inf  
def path_length(path):
    total_distance = 0
    total_distance += np.linalg.norm(start - path[0])
    for i in range(len(path) - 1):
        total_distance += np.linalg.norm(path[i] - path[i + 1])
    total_distance += np.linalg.norm(path[-1] - target)
    return total_distance
plt.ion()  
for i in range(iteration_num):
    print('第{}次迭代'.format(i + 1))
    for j in range(particle_num):
        
        fitness = path_length(particles[j])
        if not check_collision(particles[j], obstacles):  
            if fitness < particles_best_fitness[j]:
                particles_best[j] = particles[j].copy()
                particles_best_fitness[j] = fitness
            if fitness < global_best_fitness:
                global_best = particles[j].copy()
                global_best_fitness = fitness
        
        r1 = np.random.rand()
        r2 = np.random.rand()
        particles_velocity[j] = w * particles_velocity[j] + c1 * r1 * (particles_best[j] - particles[j]) + c2 * r2 * (global_best - particles[j])
        particles[j] += particles_velocity[j]
        particles[j] = np.clip(particles[j], 0, map_size)
        x = np.array([start[0], *particles[j][:, 0], target[0]]) 
        y = np.array([start[1], *particles[j][:, 1], target[1]]) 
        z = np.array([start[2], *particles[j][:, 2], target[2]]) 
        t = np.linspace(0, 1, len(x)) 
        t_new = np.linspace(0, 1, 300)  
        spl_x = make_interp_spline(t, x, k=3)(t_new) 
        spl_y = make_interp_spline(t, y, k=3)(t_new) 
        spl_z = make_interp_spline(t, z, k=3)(t_new) 
        scene.plot(spl_x, spl_y, spl_z, label='Particle {}'.format(j)) 
    plt.pause(0.1)  
    scene.clear()  
    scene.set_title('PSO for UAV Trajectory Generation')  
    scene.set_xlabel('X')  
    scene.set_ylabel('Y')  
    scene.set_zlabel('Z')  
    scene.scatter(*start, color='green', label='Start')  
    scene.scatter(*target, color='red', label='Target')  
    for center, radius in obstacles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        scene.plot_surface(x, y, z, color='gray')
scene.plot([start[0], *global_best[:, 0], target[0]], 
           [start[1], *global_best[:, 1], target[1]], 
           [start[2], *global_best[:, 2], target[2]], color='blue', label='Global Best Path')
scene.legend()
plt.ioff()  
plt.show()