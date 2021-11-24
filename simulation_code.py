import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def Sim_Brownian_Motion(t):
    # store the paths of the Brownian motion
    W = np.zeros(len(t))

    sqrt_dt = np.sqrt(t[1] - t[0])
    for i in range(len(t) - 1):
        W[i + 1] = W[i] + sqrt_dt * np.random.randn()

    return W


def risk_neutral_int_elur(r0, alpha, beta, sigma, theta0, phi, eta, t, w_sim_r, w_sim_theta):
    dt = t[1] - t[0]

    theta_path = np.zeros(len(t))
    theta_path[0] = theta0

    r_path = np.zeros(len(t))
    r_path[0] = r0

    # Euler theta and r
    for i in range(len(t)):
        theta_path[i+1] = theta_path[i] + beta * (phi - theta_path[i]) * dt + eta * (w_sim_theta[i+1] - w_sim_theta[i])
        r_path[i+1] = r_path[i] + alpha * (theta_path[i] - r_path[i]) * dt + sigma * (w_sim_r[i+1] - w_sim_r[i])

    return theta_path, r_path


if __name__ == '__main__':
    T = 1
    nsteps = 100
    nsims = 1000
    r0 = 0.02
    alpha = 3
    sigma = 0.01
    theta0 = 0.03
    beta = 1
    phi = 0.05
    eta = 0.005

    t = np.linspace(0, T, nsteps)

    theta_path_set = np.zeros((nsims, nsteps))
    r_path_set = np.zeros((nsims, nsteps))

    for i in range(0,nsims):
        w_sim_r = Sim_Brownian_Motion(t)
        w_sim_theta = Sim_Brownian_Motion(t)

        theta_path_set[i:], r_path_set[i:] = risk_neutral_int_elur(r0, alpha, beta, sigma, theta0, phi, eta, t, w_sim_r, w_sim_theta)

    bond_yield = np.average(r_path_set)

    plt.plot(t, bond_yield, label='MC bond yield')
    plt.xlabel('Time')
    plt.ylabel('Yield')
    plt.legend()
    # plt.savefig('A3_Q2.jpg')