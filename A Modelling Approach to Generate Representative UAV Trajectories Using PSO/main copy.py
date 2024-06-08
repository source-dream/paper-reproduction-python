'''
Author: sourcedream
Date: 2024-06-05 22:12:01
LastEditTime: 2024-06-07 14:34:36
Description: A Modelling Approach to Generate Representative UAV Trajectories Using PSO
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 固定参数设置
obstacles_num = 10  # 障碍物数量
map_size = np.array([120, 120, 120])  # 地图大小
start = np.array([10, 10, 10])  # 起点
target = np.array([100, 100, 80])  # 目标点
particle_num = 40  # 粒子数量
particle_point_num = 2  # 粒子位置点数量
iteration_num = 200  # 迭代次数
w = 1.2  # 惯性权重 0-2之间
c1 = 2  # 个人学习因子 0-2之间
c2 = 2  # 社会学习因子 0-2之间

# 动态参数

# 构建三维场景
figure = plt.figure()  # 创建一个新的图形窗口
scene = figure.add_subplot(111, projection='3d')  # 添加一个3D子图
# 设置改进A*算法标题
scene.set_title('improve A* Algorithm for UAV Trajectory Generation')  # 设置标题
scene.set_xlabel('X')  # 设置X轴标签
scene.set_ylabel('Y')  # 设置Y轴标签
scene.set_zlabel('Z')  # 设置Z轴标签
scene.scatter(*start, color='green', label='Start')  # 绘制起点
scene.scatter(*target, color='red', label='Target')  # 绘制目标点

# 随机生成障碍物
obstacles = []  # 初始化障碍物列表
for i in range(obstacles_num):
    center = np.random.rand(3) * map_size
    radius = np.random.rand() * 30
    obstacles.append((center, radius))
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    scene.plot_surface(x, y, z, color='blue', alpha=0.8)
    # scene.plot_surface(x, y, z, color='gray', alpha=0.5)

# 碰撞检测函数
def check_collision(path, obstacles):
    for point in path:
        for center, radius in obstacles:
            if np.linalg.norm(point - center) < radius:
                return True
    return False

# 种群初始化
particles = np.random.rand(particle_num, particle_point_num, 3) * map_size  # 随机生成粒子位置
particles_velocity = np.zeros((particle_num, particle_point_num, 3))  # 初始化粒子速度
particles_best = particles.copy()  # 个体最优位置
particles_best_fitness = np.full(particle_num, np.inf)  # 个体最优适应度
global_best = particles[0].copy()  # 全局最优位置
global_best_fitness = np.inf  # 全局最优适应度

# 计算路径长度的函数
def path_length(path):
    total_distance = 0
    total_distance += np.linalg.norm(start - path[0])
    for i in range(len(path) - 1):
        total_distance += np.linalg.norm(path[i] - path[i + 1])
    total_distance += np.linalg.norm(path[-1] - target)
    return total_distance

# 迭代优化并实时显示
plt.ion()  # 打开交互模式
for i in range(iteration_num):
    print('第{}次迭代'.format(i + 1))
    for j in range(particle_num):
        # 计算适应度
        fitness = path_length(particles[j])
        if not check_collision(particles[j], obstacles):  # 检查是否碰撞
            if fitness < particles_best_fitness[j]:
                particles_best[j] = particles[j].copy()
                particles_best_fitness[j] = fitness
            if fitness < global_best_fitness:
                global_best = particles[j].copy()
                global_best_fitness = fitness
        # 更新速度
        r1 = np.random.rand()
        r2 = np.random.rand()
        particles_velocity[j] = w * particles_velocity[j] + c1 * r1 * (particles_best[j] - particles[j]) + c2 * r2 * (global_best - particles[j])
        # 更新位置
        particles[j] += particles_velocity[j]
        # 确保粒子位置在地图范围内
        particles[j] = np.clip(particles[j], 0, map_size)
        # 绘制粒子位置
        # scene.plot([start[0], *particles[j][:, 0], target[0]], 
        #            [start[1], *particles[j][:, 1], target[1]], 
        #            [start[2], *particles[j][:, 2], target[2]], label='Particle {}'.format(j))
        # Plot particle positions
        x = np.array([start[0], *particles[j][:, 0], target[0]]) # 生成路径的x坐标
        y = np.array([start[1], *particles[j][:, 1], target[1]]) # 生成路径的y坐标
        z = np.array([start[2], *particles[j][:, 2], target[2]]) # 生成路径的z坐标
        # Smooth the path using spline interpolation
        t = np.linspace(0, 1, len(x)) # 生成路径的参数t
        t_new = np.linspace(0, 1, 300)  # 生成新的参数t
        spl_x = make_interp_spline(t, x, k=3)(t_new) # 生成x坐标的插值
        spl_y = make_interp_spline(t, y, k=3)(t_new) # 生成y坐标的插值
        spl_z = make_interp_spline(t, z, k=3)(t_new) # 生成z坐标的插值
        scene.plot(spl_x, spl_y, spl_z, label='Particle {}'.format(j)) # 绘制路径
    plt.pause(0.1)  # 暂停0.1秒
    scene.clear()  # 清空子图
    scene.set_title('improve A* Algorithm for UAV Trajectory Generation')  # 设置标题
    scene.set_xlabel('X')  # 重新设置X轴标签
    scene.set_ylabel('Y')  # 重新设置Y轴标签
    scene.set_zlabel('Z')  # 重新设置Z轴标签
    scene.scatter(*start, color='green', label='Start')  # 重新绘制起点
    scene.scatter(*target, color='red', label='Target')  # 重新绘制目标点
    # 重新绘制障碍物
    for center, radius in obstacles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        scene.plot_surface(x, y, z, color='pink', alpha=0.8)

# 绘制最终的全局最优路径
scene.plot([start[0], *global_best[:, 0], target[0]], 
           [start[1], *global_best[:, 1], target[1]], 
           [start[2], *global_best[:, 2], target[2]], color='blue', label='Global Best Path')
scene.legend()
plt.ioff()  # 关闭交互模式
plt.show()