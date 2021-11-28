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


def bond_price_sim(r_path, t):
    dt = t[1] - t[0]
    bank_account = np.zeros(len(t)+1)
    bank_account[0] = 1
    for i in range(0, len(t)):
        bank_account[i+1] = bank_account[i] + r_path[i] * bank_account[i] * dt
    bond_price = [1 / b for b in bank_account]
    return bond_price[1:]


def bond_yield_sim(int_matrix, nsims, nsteps, t):
    dt = t[1] - t[0]
    # interest rate to price
    mc_bond_price = np.empty([nsims,nsteps])
    for n in range(0, nsims):
        mc_bond_price[n] = bond_price_sim(int_matrix[n], t)
    # price_std = np.std(mc_bond_price)
    mc_bond_price = np.matrix(mc_bond_price)
    mc_bond_price_avg = np.asarray(mc_bond_price.mean(0)).reshape(-1)
    mc_bond_price_std = np.asarray(mc_bond_price.std(0)).reshape(-1)

    mc_bond_price_upper = np.zeros(len(t))
    mc_bond_price_lower = np.zeros(len(t))
    for i in range(0, nsteps):
        ci = 3 * mc_bond_price_std[i]/np.sqrt(len(t))
        mc_bond_price_upper[i] = mc_bond_price_avg[i] + ci
        mc_bond_price_lower[i] = mc_bond_price_avg[i] - ci


    mc_bond_yield = np.zeros(len(t))
    # mc_bond_yield[0] = r0
    mc_bond_yield_upper = np.zeros(len(t))
    # mc_bond_yield_upper[0] = r0
    mc_bond_yield_lower = np.zeros(len(t))
    # mc_bond_yield_lower[0] = r0

    for i in range(0, nsteps):
        mc_bond_yield[i] = - np.log(mc_bond_price_avg[i]) / (dt * (i + 1))
        mc_bond_yield_upper[i] = - np.log(mc_bond_price_upper[i]) / (dt * (i + 1))
        mc_bond_yield_lower[i] = - np.log(mc_bond_price_lower[i]) / (dt * (i + 1))
    mc_bond_yield_lower = mc_bond_yield - mc_bond_yield_lower
    mc_bond_yield_upper = mc_bond_yield_upper - mc_bond_yield
    return mc_bond_yield, mc_bond_yield_upper, mc_bond_yield_lower


def analytic_a(alpha, beta, sigma, phi, eta, T, t):
    m = e ** -(alpha * (T - t))
    n = e ** -(beta * (T - t))
    p1 = - phi * (T - t)
    p2 = - phi * beta * (1 - m) / (alpha * (alpha - beta))
    p3 = phi * alpha * (1 - n) / (beta * (alpha - beta))
    pb = sigma ** 2 / (2 * alpha ** 2) * (T - t) \
        - sigma ** 2 * (1- m) / alpha ** 3 \
        + sigma ** 2 * (1 - m ** 2) / (4 * alpha ** 3)
    pc = eta ** 2 / (2 * beta ** 2) * (
        (T - t)
        + 2 * beta * (1 - m) / (alpha * (alpha - beta))
        - 2 * alpha * (1 - n) / (beta * (alpha - beta))
        + beta ** 2 * (1 - m ** 2) / (2 * alpha * (alpha - beta) ** 2)
        - 2 * alpha * beta * (1 - m * n) / ((alpha + beta) * (alpha - beta) ** 2)
        + alpha ** 2 * (1 - n ** 2) / (2 * beta * (alpha - beta) ** 2)
    )

    return p1 + p2 + p3 + pb + pc


def analytic_b(alpha, T, t):
    return (1 - e ** -(alpha * (T - t))) / alpha


def analytic_c(alpha, beta, T, t):
    return (1 / beta
            + (beta * e ** -(alpha * (T - t)) - alpha * e ** -(beta * (T - t))) / (beta * (alpha - beta)))


def analytic_formula_curve(r0, alpha, beta, sigma, theta0, phi, eta, T, t):
    return (analytic_b(alpha, T, t) * r0
            + analytic_c(alpha, beta, T, t) * theta0
            - analytic_a(alpha, beta, sigma, phi, eta, T, t)) / (T - t)



if __name__ == '__main__':
    T = 10
    nsteps = 40
    nsims = 1000
    r0 = 0.02
    alpha = 3
    sigma = 0.01
    theta0 = 0.03
    beta = 1
    phi = 0.05
    eta = 0.005

    t = np.linspace(0, T, nsteps)

    # MC simulation
    theta_path_set = np.zeros((nsims, nsteps))
    r_path_set = np.zeros((nsims, nsteps))

    # simulate interest rate
    for i in range(0, nsims):
        w_sim_r = Sim_Brownian_Motion(t)
        w_sim_theta = Sim_Brownian_Motion(t)
        [theta_path_set[i:], r_path_set[i:]] = risk_neutral_int_elur(r0, alpha, beta, sigma, theta0, phi, eta, t, w_sim_r, w_sim_theta)
    int_matrix = r_path_set.reshape(nsims, nsteps)
    # r_path_set_avg = np.asarray(r_path_set_avg).reshape(-1)

    # simulate yield curve
    mc_bond_yield, mc_bond_yield_upper, mc_bond_yield_lower = bond_yield_sim(int_matrix, nsims, nsteps, t)

    # analytic formula
    analytic_bond_yield = np.zeros(nsteps)
    analytic_bond_yield[0] = r0
    for i in range(1, nsteps):
        analytic_bond_yield[i] = analytic_formula_curve(r0, alpha, beta, sigma, theta0, phi, eta, t[i], t=0)

    # analytic vs mc
    plt.figure(1)
    plt.errorbar(t, mc_bond_yield, yerr=[mc_bond_yield_lower,mc_bond_yield_upper], fmt='-', elinewidth=0.5, capsize=2, label='MC bond yield with error band')
    plt.plot(t, analytic_bond_yield, label='Analytic bond yield')
    plt.xlabel('Time')
    plt.ylabel('Yield')
    plt.title('Analytic vs Monte Carlo Simulation  Bond Yield Comparison')
    plt.legend()
    plt.savefig('A3_Q2_MCvsAna.jpg')

    # mean reverting graphs
    w_sim_r_for_graph = Sim_Brownian_Motion(np.linspace(0, T, nsims))
    w_sim_theta_for_graph = Sim_Brownian_Motion(np.linspace(0, T, nsims))
    [theta_path_set_for_graph, r_path_set_for_graph] = risk_neutral_int_elur(r0, alpha, beta, sigma, theta0, phi, eta, np.linspace(0, T, nsims), w_sim_r_for_graph,
                                                                 w_sim_theta_for_graph)
    plt.figure(2)
    plt.plot(np.linspace(0, T, nsims), theta_path_set_for_graph, label='Long Run Interest Rate')
    plt.plot(np.linspace(0, T, nsims), r_path_set_for_graph, label='Short Run Interest Rate')
    plt.plot(np.linspace(0, T, nsims), np.ones(nsims)*phi, label='Mean Reverting Level (\u03C6)')
    plt.xlabel('Time')
    plt.ylabel('Rate')
    plt.title('Stochastic Interest Rate Simulation')
    plt.legend()
    plt.savefig('A3_Q2_MeanRevert.jpg')

