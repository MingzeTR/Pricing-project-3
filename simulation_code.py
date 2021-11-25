import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import e


def Sim_Brownian_Motion(t):
    # store the paths of the Brownian motion
    W = np.zeros(len(t+1))

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
    for i in range(len(t) - 1):
        theta_path[i+1] = theta_path[i] + beta * (phi - theta_path[i]) * dt + eta * (w_sim_theta[i+1] - w_sim_theta[i])
        r_path[i+1] = r_path[i] + alpha * (theta_path[i] - r_path[i]) * dt + sigma * (w_sim_r[i+1] - w_sim_r[i])

    return theta_path, r_path


def analytic_a(alpha, beta, sigma, phi, eta, T, t):
    m = alpha * (T - t)
    n = beta * (T - t)
    p1 = - phi * (T - t)
    p2 = - phi * beta * (1 - m) / (alpha * (alpha - beta))
    p3 = phi * alpha * (1 - n) / (beta * (alpha - beta))
    pb = sigma ** 2 / (2 * alpha ** 2) * (T - t) \
        - sigma ** 2 * (1- m) / alpha ** 3 \
        + (1 - m ** 2) / (4 * alpha ** 3)
    pc = eta ** 2 / (2 * beta ** 2) * (
        (T - t)
        + 2 * beta * (1 - m) / (alpha * (alpha - beta))
        - 2 * alpha * (1 - n) / (beta * (alpha - beta))
        + beta ** 2 * (1 - m ** 2) / (2 * alpha * (alpha - beta) ** 2)
        - 2 * alpha * beta * (1 - m * n) / ((alpha + beta) * (alpha - beta) ** 2)
        + alpha ** 2 *(1 - n ** 2) / (2 * beta * (alpha - beta) ** 2)
    )

    return p1 + p2 + p3 + pb + pc


def analytic_b(alpha, T, t):
    return (1 - e ** -(alpha * (T - t))) / alpha


def analytic_c(alpha, beta, T, t):
    return ((e ** -(alpha * (T - t)) - 1) / (alpha - beta)
            + alpha * (1 - e ** -(beta * (T - t))) / ((alpha - beta) * beta))


def analytic_formula_curve(r0, alpha, beta, sigma, theta0, phi, eta, T, t):
    return (analytic_c(alpha, beta, T, t) * r0
            + analytic_c(alpha, beta, T, t) * theta0
            - analytic_a(alpha, beta, sigma, phi, eta, T, t)) / (T - t)



if __name__ == '__main__':
    T = 10
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

    for i in range(0, nsims):
        w_sim_r = Sim_Brownian_Motion(t)
        w_sim_theta = Sim_Brownian_Motion(t)

        [theta_path_set[i:], r_path_set[i:]] = risk_neutral_int_elur(r0, alpha, beta, sigma, theta0, phi, eta, t, w_sim_r, w_sim_theta)

    mc_bond_yield = int_matrix = np.matrix(r_path_set.reshape(1000,100)).mean(0)

    analytic_bond_yield = np.zeros(nsteps)
    for i in range(1, nsteps):
        analytic_bond_yield[i] = analytic_formula_curve(r0, alpha, beta, sigma, theta0, phi, eta, t[i], t=0)

    plt.plot(t, np.asarray(mc_bond_yield).reshape(-1), label='MC bond yield')
    plt.plot(t, analytic_bond_yield, label='Analytic bond yield')
    plt.xlabel('Time')
    plt.ylabel('Yield')
    plt.legend()
    # plt.savefig('A3_Q2.jpg')