#!/usr/bin/env python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import csv
import torch  
from milo import milo 
from novograd_torch import NovoGrad

# Import configuration
from config import OUTPUT_DIR, RUNS, GRID_POINTS, DEFAULT_OPTIMIZER_PARAMS, BENCHMARKS, PLOT_SETTINGS

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

#############################################
# Numerical Gradient Helper
#############################################
def numerical_gradient(f, x, h=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

#############################################
# Benchmark Functions (Numpy) and Torch Versions
#############################################

# 1. Himmelblau (analytic gradient)
def himmelblau(x):
    x1, x2 = x[0], x[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

# Torch version of Himmelblau using torch operations
def torch_himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def grad_himmelblau(x):
    x1, x2 = x[0], x[1]
    u = x1**2 + x2 - 11
    v = x1 + x2**2 - 7
    grad_x = 4 * x1 * u + 2 * v
    grad_y = 2 * u + 4 * x2 * v
    return np.array([grad_x, grad_y])

# 2. Eggholder (numerical gradient)
def egg(x):
    x1, x2 = x[0], x[1]
    term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1/2 + 47)))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    return term1 + term2

#############################################
# Add torch version of Eggholder function
#############################################
def torch_egg(x):
    x1 = x[0]
    x2 = x[1]
    term1 = -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x2 + x1/2 + 47)))
    term2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47))))
    return term1 + term2

# 3. Ackley (numerical gradient)
def ackley(x, a=20, b=0.2, c=2*np.pi):
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

# 4. Camel6 (numerical gradient)
def camel6(x):
    x1, x2 = x[0], x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    return term1 + term2 + term3

# 5. De Jong5 (numerical gradient)
def dejong5(x):
    x1, x2 = x[0], x[1]
    s = 0.0
    a_vals = np.array([-32, -16, 0, 16, 32])
    A1 = np.tile(a_vals, 5)
    A2 = np.repeat(a_vals, 5)
    for i in range(25):
        a1i = A1[i]
        a2i = A2[i]
        term1 = i + 1  # MATLAB indexing: 1...25
        term2 = (x1 - a1i)**6
        term3 = (x2 - a2i)**6
        s += 1.0 / (term1 + term2 + term3)
    return 1.0 / (0.002 + s)

# 6. Michalewicz (numerical gradient)
def michal(x, m=10):
    d = len(x)
    s = 0.0
    for i in range(d):
        xi = x[i]
        s += np.sin(xi) * (np.sin((i+1) * xi**2 / np.pi))**(2*m)
    return -s

#############################################
# Clamp helper functions
#############################################
def clamp_np(point, xmin, xmax, ymin, ymax):
    return np.array([np.clip(point[0], xmin, xmax), np.clip(point[1], ymin, ymax)])

#############################################
# Modified Torch-based Optimization Runner with clamping
#############################################
def run_optimization_pt(torch_f, init_point, domain, sgd_params=None, adam_params=None, milo_params=None, novograd_params=None):
    # Set default parameters if not provided
    if sgd_params is None:
        sgd_params = DEFAULT_OPTIMIZER_PARAMS["SGD"]
    if adam_params is None:
        adam_params = DEFAULT_OPTIMIZER_PARAMS["ADAM"]
    if milo_params is None:
        milo_params = DEFAULT_OPTIMIZER_PARAMS["milo"]
    if novograd_params is None:
        novograd_params = DEFAULT_OPTIMIZER_PARAMS["NOVOGRAD"]
    
    xmin, xmax, ymin, ymax = domain

    def run_optimizer(optimizer_class, iterations, **kwargs):
        param = torch.tensor(init_point, dtype=torch.float32, requires_grad=True)
        optimizer = optimizer_class([param], **kwargs)
        traj = [param.detach().cpu().numpy().copy()]
        for _ in range(iterations):
            optimizer.zero_grad()
            loss = torch_f(param)
            loss.backward()
            # For Adam or milo, inject noise if gradient norm is small to avoid getting stuck
            if optimizer_class == torch.optim.Adam or optimizer_class == milo:
                grad_norm = param.grad.data.norm()
                if grad_norm < 1e-3:
                    param.grad.data.add_(torch.randn_like(param.grad.data) * kwargs.get("lr", 0.1) * 0.5)
            optimizer.step()
            # Clamp the parameter values to remain within domain bounds
            with torch.no_grad():
                param[0] = torch.clamp(param[0], xmin, xmax)
                param[1] = torch.clamp(param[1], ymin, ymax)
            traj.append(param.detach().cpu().numpy().copy())
        return np.array(traj)

    def custom_sgd_update(param, grad, lr, momentum, velocity, grad_clip=1.0):
        # Gradient Clipping
        grad_norm = torch.linalg.norm(grad)
        if grad_norm > grad_clip:
            grad = grad_clip * (grad / grad_norm)

        velocity = momentum * velocity + lr * grad
        param.data -= velocity
        return param, velocity

    def run_custom_sgd(init_point, lr, momentum=0.0, iterations=100, grad_clip=1.0):
        param = torch.tensor(init_point, dtype=torch.float32, requires_grad=True)
        velocity = torch.zeros_like(param)
        traj = [param.detach().cpu().numpy().copy()]

        for _ in range(iterations):
            loss = torch_f(param)
            loss.backward()

            with torch.no_grad():
                param, velocity = custom_sgd_update(param, param.grad, lr, momentum, velocity, grad_clip)
                param[0] = torch.clamp(param[0], xmin, xmax)
                param[1] = torch.clamp(param[1], ymin, ymax)
                param.grad.zero_()

            traj.append(param.detach().cpu().numpy().copy())
        return np.array(traj)

    # Extract parameters for each optimizer
    sgd_eta = sgd_params.get("eta", 0.1)
    sgd_iterations = sgd_params.get("iterations", 100)
    sgd_momentum = sgd_params.get("momentum", 0.009)
    sgd_grad_clip = sgd_params.get("grad_clip", 1.0)
    
    adam_eta = adam_params.get("eta", 0.1)
    adam_iterations = adam_params.get("iterations", 100)
    
    milo_eta = milo_params.get("eta", 0.1)
    milo_iterations = milo_params.get("iterations", 100)
    
    novograd_eta = novograd_params.get("eta", 0.1)
    novograd_iterations = novograd_params.get("iterations", 100)
    
    # Run optimizers with their specific parameters
    traj_sgd = run_custom_sgd(init_point, lr=sgd_eta, momentum=sgd_momentum, iterations=sgd_iterations, grad_clip=sgd_grad_clip)
    traj_adam = run_optimizer(torch.optim.Adam, adam_iterations, lr=adam_eta, amsgrad=adam_params.get("amsgrad", True))
    traj_milo = run_optimizer(milo, milo_iterations, lr=milo_eta,
                            group_size=milo_params.get("group_size", 150),
                            momentum=milo_params.get("momentum", 0.30),
                            adaptive=milo_params.get("adaptive", True),
                            adaptive_eps=milo_params.get("adaptive_eps", 1e-8),
                            weight_decay=milo_params.get("weight_decay", 0.05),
                            layer_wise=milo_params.get("layer_wise", False))
    traj_novograd = run_optimizer(NovoGrad, novograd_iterations, lr=novograd_eta,
                                  grad_averaging=novograd_params.get("grad_averaging", True))
    return traj_sgd, traj_adam, traj_milo, traj_novograd

#############################################
# Torch versions for remaining benchmarks
#############################################
import math

def torch_ackley(x, a=20, b=0.2, c=2*math.pi):
    d = x.shape[0]
    sum1 = torch.sum(x**2)
    sum2 = torch.sum(torch.cos(torch.tensor(c, dtype=x.dtype)*x))
    term1 = -a * torch.exp(-b * torch.sqrt(sum1 / d))
    term2 = -torch.exp(sum2 / d)
    return term1 + term2 + a + torch.exp(torch.tensor(1.0, dtype=x.dtype))

def torch_camel6(x):
    x1 = x[0]
    x2 = x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * (x1**2)
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * (x2**2)
    return term1 + term2 + term3

def torch_dejong5(x):
    x1 = x[0]
    x2 = x[1]
    s = torch.tensor(0.0, dtype=x.dtype)
    a_vals = torch.tensor([-32, -16, 0, 16, 32], dtype=x.dtype)
    A1 = a_vals.repeat(5)
    A2 = a_vals.repeat_interleave(5)
    for i in range(25):
        a1i = A1[i]
        a2i = A2[i]
        term1 = i + 1  # 1-indexed
        term2 = (x1 - a1i)**6
        term3 = (x2 - a2i)**6
        s = s + 1.0 / (term1 + term2 + term3)
    return 1.0 / (0.002 + s)

def torch_michal(x, m=10):
    d = x.shape[0]
    s = torch.tensor(0.0, dtype=x.dtype)
    for i in range(d):
        xi = x[i]
        s = s + torch.sin(xi) * (torch.sin((i+1) * xi**2 / torch.tensor(math.pi, dtype=x.dtype)))**(2*m)
    return -s

#############################################
# Plotting Routine for Each Benchmark (Updated)
#############################################
sns.set(style="whitegrid")

for bm in BENCHMARKS:
    name = bm["name"]
    # Get the corresponding function from this module based on the name
    numpy_f_name = name.lower()
    if numpy_f_name == "de jong5":
        numpy_f_name = "dejong5"
    elif numpy_f_name == "michalewicz":
        numpy_f_name = "michal"
    
    numpy_f = globals()[numpy_f_name]
    torch_f_name = "torch_" + numpy_f_name
    torch_f = globals()[torch_f_name]
    
    if torch_f is None:
        raise ValueError(f"Benchmark {name} must define a torch function using torch operations.")
    
    xmin, xmax, ymin, ymax = bm["domain"]
    domain = bm["domain"]
    minima = bm["minima"]
    
    # Get initial point
    init_point = list(bm.get("init_point", [np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)]))
    # Ensure the same initial point is used across all runs
    base_init_point = init_point.copy()
    
    # --- Run each optimizer multiple times using the same initial point and average trajectories ---
    all_trajs_sgd, all_trajs_adam, all_trajs_milo, all_trajs_novograd = [], [], [], []
    elapsed_total = 0
    metrics = [] # re-initialize metrics list for each benchmark
    for _ in range(RUNS):
        # Pass the optimizer-specific parameters
        traj_sgd, traj_adam, traj_milo, traj_novograd = run_optimization_pt(
            torch_f, base_init_point, domain,
            sgd_params=bm.get("sgd_params"),
            adam_params=bm.get("adam_params"),
            milo_params=bm.get("milo_params"),
            novograd_params=bm.get("novograd_params")
        )
        elapsed_total += time.perf_counter() - start_time if 'start_time' in locals() else 0
        all_trajs_sgd.append(traj_sgd)
        all_trajs_adam.append(traj_adam)
        all_trajs_milo.append(traj_milo)
        all_trajs_novograd.append(traj_novograd)
        start_time = time.perf_counter()  # start timing for subsequent iterations
    
    avg_traj_sgd = np.mean(all_trajs_sgd, axis=0)
    avg_traj_adam = np.mean(all_trajs_adam, axis=0)
    avg_traj_milo = np.mean(all_trajs_milo, axis=0)
    avg_traj_novograd = np.mean(all_trajs_novograd, axis=0)
    avg_elapsed = elapsed_total / RUNS
    
    # Compute final metrics using averaged trajectory
    def final_metrics(traj):
        final_point = traj[-1]
        final_loss = numpy_f(final_point)  # using numpy f
        distances = [np.linalg.norm(final_point - np.array(loc)) for loc in minima]
        return final_point, final_loss, min(distances) if distances else None

    sgd_final, sgd_loss, sgd_distance = final_metrics(avg_traj_sgd)
    adam_final, adam_loss, adam_distance = final_metrics(avg_traj_adam)
    milo_final, milo_loss, milo_distance = final_metrics(avg_traj_milo)
    novograd_final, novograd_loss, novograd_distance = final_metrics(avg_traj_novograd)
    
    # Create mesh grid for plotting
    x_vals = np.linspace(xmin, xmax, GRID_POINTS)
    y_vals = np.linspace(ymin, ymax, GRID_POINTS)
    X, Y = np.meshgrid(x_vals, y_vals)
    f_vec = np.vectorize(lambda x, y: numpy_f(np.array([x, y])))
    Z = f_vec(X, Y)
    
    # For 3D plots, compute an offset based on 1% of the z-range
    z_min, z_max = np.min(Z), np.max(Z)
    z_offset = 0.01 * (z_max - z_min)
    
    # -------------------------------
    # 2D Contour Plot (using averaged trajectories)
    # -------------------------------
    plt.figure(figsize=PLOT_SETTINGS["2D"]["figsize"])
    contour = plt.contourf(X, Y, Z, levels=PLOT_SETTINGS["2D"]["levels"], cmap=PLOT_SETTINGS["2D"]["cmap"])
    #plt.colorbar(contour)
    plt.contour(X, Y, Z, levels=PLOT_SETTINGS["2D"]["levels"], 
                colors=PLOT_SETTINGS["2D"]["contour_colors"], 
                linewidths=PLOT_SETTINGS["2D"]["contour_linewidths"], 
                alpha=PLOT_SETTINGS["2D"]["contour_alpha"])
    
    # Plot trajectories
    plt.plot(avg_traj_sgd[:,0], avg_traj_sgd[:,1], PLOT_SETTINGS["markers"]["SGD"]["line"], label=PLOT_SETTINGS["markers"]["SGD"]["label"], zorder=10)
    plt.plot(avg_traj_adam[:,0], avg_traj_adam[:,1], PLOT_SETTINGS["markers"]["ADAM"]["line"], label=PLOT_SETTINGS["markers"]["ADAM"]["label"], zorder=10)
    plt.plot(avg_traj_milo[:,0], avg_traj_milo[:,1], PLOT_SETTINGS["markers"]["milo"]["line"], label=PLOT_SETTINGS["markers"]["milo"]["label"], zorder=10)
    plt.plot(avg_traj_novograd[:,0], avg_traj_novograd[:,1], PLOT_SETTINGS["markers"]["NOVOGRAD"]["line"], label=PLOT_SETTINGS["markers"]["NOVOGRAD"]["label"], zorder=10)
    
    # Plot points
    plt.plot(init_point[0], init_point[1], PLOT_SETTINGS["markers"]["start"]["marker"], markersize=PLOT_SETTINGS["markers"]["start"]["size"], label=PLOT_SETTINGS["markers"]["start"]["label"], zorder=20)
    plt.plot(sgd_final[0], sgd_final[1], PLOT_SETTINGS["markers"]["SGD"]["marker"], markersize=12, markerfacecolor='none', label='SGD Final', zorder=20)
    plt.plot(adam_final[0], adam_final[1], PLOT_SETTINGS["markers"]["ADAM"]["marker"], markersize=12, markerfacecolor='none', label='ADAM Final', zorder=20)
    plt.plot(milo_final[0], milo_final[1], PLOT_SETTINGS["markers"]["milo"]["marker"], markersize=12, markerfacecolor='none', label='milo Final', zorder=20)
    plt.plot(novograd_final[0], novograd_final[1], PLOT_SETTINGS["markers"]["NOVOGRAD"]["marker"], markersize=12, markerfacecolor='none', label='NovoGrad Final', zorder=20)
    
    for loc in minima:
        plt.plot(loc[0], loc[1], PLOT_SETTINGS["markers"]["minima"]["marker"], markersize=PLOT_SETTINGS["markers"]["minima"]["size"], label=PLOT_SETTINGS["markers"]["minima"]["label"], zorder=20)
    
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title(f"{name}: 2D Contour with Averaged Optimization Trajectories\n(Random init = {np.array(init_point).round(2)})")
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_2d_contour.jpeg"), format='jpeg', dpi=300)
    #plt.show()
    
    # -------------------------------
    # 3D Surface Plot (using averaged trajectories)
    # -------------------------------
    fig = plt.figure(figsize=PLOT_SETTINGS["3D"]["figsize"])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=PLOT_SETTINGS["3D"]["cmap"], alpha=PLOT_SETTINGS["3D"]["surface_alpha"])
    
    # Compute loss values for each trajectory
    sgd_loss_vals = np.array([numpy_f(p) for p in avg_traj_sgd])
    adam_loss_vals = np.array([numpy_f(p) for p in avg_traj_adam])
    milo_loss_vals = np.array([numpy_f(p) for p in avg_traj_milo])
    novograd_loss_vals = np.array([numpy_f(p) for p in avg_traj_novograd])
    
    # Plot trajectories in 3D
    ax.plot(avg_traj_sgd[:,0], avg_traj_sgd[:,1], sgd_loss_vals, PLOT_SETTINGS["markers"]["SGD"]["line"], label='SGD Trajectory', linewidth=2)
    ax.plot(avg_traj_adam[:,0], avg_traj_adam[:,1], adam_loss_vals, PLOT_SETTINGS["markers"]["ADAM"]["line"], label='ADAM Trajectory', linewidth=2)
    ax.plot(avg_traj_milo[:,0], avg_traj_milo[:,1], milo_loss_vals, PLOT_SETTINGS["markers"]["milo"]["line"], label='milo Trajectory', linewidth=2)
    ax.plot(avg_traj_novograd[:,0], avg_traj_novograd[:,1], novograd_loss_vals, PLOT_SETTINGS["markers"]["NOVOGRAD"]["line"], label='NovoGrad Trajectory', linewidth=2)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Loss')
    # ax.set_title("3D Surface with Averaged Loss Trajectories")
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_3d_loss_map.jpeg"), format='jpeg', dpi=300)
    #plt.show()

#############################################
# Write Metrics to CSV File
#############################################
csv_file = os.path.join(OUTPUT_DIR, "metrics.csv")
with open(csv_file, mode='w', newline='') as f_csv:
    writer = csv.writer(f_csv)
    header = ["Function", "Optimizer", "Initial Point", "Final Point", "Final Loss", "Distance to Ideal", "Time (s)", "Iterations"]
    writer.writerow(header)
    for row in metrics:
        writer.writerow([
            row["Function"],
            row["Optimizer"],
            np.array2string(np.array(row["Initial Point"]), precision=2),
            np.array2string(np.array(row["Final Point"]), precision=2),
            row["Final Loss"],
            row["Distance to Ideal"],
            row["Time (s)"],
            row["Iterations"]
        ])

print(f"Metrics written to {csv_file}")
