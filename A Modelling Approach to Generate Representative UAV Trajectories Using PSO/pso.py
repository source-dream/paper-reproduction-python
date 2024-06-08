'''
Author: sourcedream
Date: 2024-06-06 14:10:06
LastEditTime: 2024-06-06 21:43:50
Description: 粒子群算法实现
'''
import numpy as np
from scipy.interpolate import CubicSpline
from pyswarm import pso
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成轨迹函数
def generate_trajectory(waypoints):
    t = np.linspace(0, 1, len(waypoints))
    cs = CubicSpline(t, waypoints, bc_type='clamped')
    return cs

# 轨迹代价函数
def trajectory_cost_function(params, waypoints, obstacles):
    params = params.reshape((-1, 3))
    t = np.linspace(0, 1, len(params))
    cs = CubicSpline(t, params, bc_type='clamped')
    t_wp = np.linspace(0, 1, len(waypoints))
    generated_wp = cs(t_wp)
    cost = np.sum((generated_wp - waypoints) ** 2)

    # 避障成本
    for obs in obstacles:
        dist = np.linalg.norm(generated_wp - obs, axis=1)
        cost += np.sum(np.exp(-dist))
    
    return cost

# 粒子群优化函数
def pso_optimization(waypoints, obstacles):
    lb = [-10] * 30
    ub = [10] * 30
    xopt, fopt = pso(trajectory_cost_function, lb, ub, args=(waypoints, obstacles))
    return xopt

# 初始化路径点和障碍物
waypoints = np.array([[0, 0, 0], [10, 10, 10]])
obstacles = [np.array([5, 5, 5]), np.array([7, 7, 7])]

# PSO优化
optimal_params = pso_optimization(waypoints, obstacles)
optimal_params = optimal_params.reshape((-1, 3))
optimal_trajectory = generate_trajectory(optimal_params)

# 可视化优化后的轨迹
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
t = np.linspace(0, 1, 100)
trajectory_points = optimal_trajectory(t)
ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2])
ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='r')
for obs in obstacles:
    ax.scatter(obs[0], obs[1], obs[2], c='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()