import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.interpolate import make_interp_spline
from matplotlib.colors import ListedColormap

# 固定参数设置
obstacles_num = 5  # 障碍物数量
map_size = np.array([100, 100, 100])  # 地图大小
start = np.array([0, 0, 0])  # 起点
target = np.array([100, 100, 80])  # 目标点
particle_num = 50  # 粒子数量
particle_point_num = 3  # 粒子位置点数量
iteration_num = 100  # 迭代次数
w = 1.2  # 惯性权重 0-2之间
c1 = 2  # 个人学习因子 0-2之间
c2 = 3  # 社会学习因子 0-2之间

# 构建三维场景
figure = plt.figure()  # 创建一个新的图形窗口
scene = figure.add_subplot(111, projection='3d')  # 添加一个3D子图
scene.set_title('PSO for UAV Trajectory Generation')  # 设置标题
scene.set_xlabel('X')  # 设置X轴标签
scene.set_ylabel('Y')  # 设置Y轴标签
scene.set_zlabel('Z')  # 设置Z轴标签
scene.scatter(*start, color='green', label='Start')  # 绘制起点
scene.scatter(*target, color='red', label='Target')  # 绘制目标点

# 随机生成障碍物
obstacles = []  # 初始化障碍物列表
for i in range(obstacles_num):
    center = np.random.rand(3) * map_size
    radius = np.random.rand() * 50
    obstacles.append((center, radius))
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    scene.plot_surface(x, y, z, color='gray', alpha=0.5)

# 检查路径是否与障碍物相交
def check_collision(path, obstacles):
    for point in path:
        for center, radius in obstacles:
            if np.linalg.norm(point - center) < radius:
                return True
    return False

# 种群初始化
particles = np.random.rand(particle_num, particle_point_num, 3) * map_size  # 随机生成粒子和粒子位置点
print(particles)
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

# 获取平滑的路径
def smooth_path(path):
    x = np.append([start[0]], np.append(path[:, 0], [target[0]]))
    y = np.append([start[1]], np.append(path[:, 1], [target[1]]))
    z = np.append([start[2]], np.append(path[:, 2], [target[2]]))
    
    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, 300)
    
    spline_x = make_interp_spline(t, x, k=3)
    spline_y = make_interp_spline(t, y, k=3)
    spline_z = make_interp_spline(t, z, k=3)
    
    smooth_x = spline_x(t_new)
    smooth_y = spline_y(t_new)
    smooth_z = spline_z(t_new)
    
    return smooth_x, smooth_y, smooth_z

# 迭代优化并实时显示
plt.ion()  # 打开交互模式
for i in range(iteration_num):
    print('第{}次迭代'.format(i + 1))
    for j in range(particle_num):
        # 计算适应度
        fitness = path_length(particles[j])
        if not check_collision(particles[j], obstacles) and fitness < particles_best_fitness[j]:
            particles_best[j] = particles[j].copy()
            particles_best_fitness[j] = fitness
        if not check_collision(particles[j], obstacles) and fitness < global_best_fitness:
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
        smooth_x, smooth_y, smooth_z = smooth_path(particles[j])
        scene.plot(smooth_x, smooth_y, smooth_z, label='Particle {}'.format(j))
    plt.pause(0.1)  # 暂停0.1秒
    scene.clear()  # 清空子图
    scene.set_title('PSO for UAV Trajectory Generation')  # 重新设置标题
    scene.set_xlabel('X')  # 重新设置X轴标签
    scene.set_ylabel('Y')  # 重新设置Y轴标签
    scene.set_zlabel('Z')  # 重新设置Z轴标签
    scene.scatter(*start, color='green', label='Start')  # 重新绘制起点
    scene.scatter(*target, color='red', label='Target')  # 重新绘制目标点
    for center, radius in obstacles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        scene.plot_surface(x, y, z, color='gray', alpha=0.5)

# 绘制最终的全局最优路径
smooth_x, smooth_y, smooth_z = smooth_path(global_best)
scene.plot(smooth_x, smooth_y, smooth_z, color='blue', label='Global Best Path')
scene.legend()
plt.ioff()  # 关闭交互模式
plt.show()