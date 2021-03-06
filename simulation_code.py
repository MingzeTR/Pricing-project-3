import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import e
import argparse
import pandas as pd

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


def forward_neutral_int_elur(T1, T2, r0, alpha, beta, sigma, theta0, phi, eta, t, w_sim_r, w_sim_theta):
    dt = t[1] - t[0]

    theta_path_for = np.zeros(len(t))
    theta_path_for[0] = theta0

    r_path_for = np.zeros(len(t))
    r_path_for[0] = r0

    # Euler theta and r
    for i in range(len(t) - 1):
        B_T1 = analytic_b(alpha, T1, t[i])
        B_T2 = analytic_b(alpha, T2, t[i])
        C_T1 = analytic_c(alpha, beta, T1, t[i])
        C_T2 = analytic_c(alpha, beta, T2, t[i])

        lambda1 = sigma * (B_T1 ** 2) / (B_T1 - B_T2)
        lambda2 = eta * (C_T1 ** 2) / (C_T1 - C_T2)

        theta_path_for[i + 1] = theta_path_for[i] + (beta * (phi - theta_path_for[i]) - lambda2) * dt + eta * (
                    w_sim_theta[i + 1] - w_sim_theta[i])
        r_path_for[i + 1] = r_path_for[i] + (alpha * (theta_path_for[i] - r_path_for[i]) - lambda1) * dt + sigma * (
                    w_sim_r[i + 1] - w_sim_r[i])

    return theta_path_for, r_path_for


def bond_price(t1, t2, t, step, r_path):
    dt = t[1] - t[0]
    bank_account = np.zeros(int(t2 * step)+1)
    bank_account[0] = 1
    for i in range(0, int(t2 * step)):
        bank_account[i+1] = bank_account[i] + r_path[i] * bank_account[i] * dt
    return bank_account[t1 * step] / bank_account[int(t2 * step)]


def bond_price_sim(r_path, t):
    dt = t[1] - t[0]
    bank_account = np.zeros(len(t)+1)
    bank_account[0] = 1
    for i in range(0, len(t)):
        bank_account[i+1] = bank_account[i] + r_path[i] * bank_account[i] * dt
    bond_price = [1 / b for b in bank_account]
    return bond_price[1:]

def bond_price_forward(t1, t2, t, step, r_path, theta_path, alpha, beta, sigma, phi, eta):
    dt = t[1] - t[0]
    price_list = np.zeros(int(t2 * step)+1)
    price_list[0] = e ** (
        (analytic_a(alpha, beta, sigma, phi, eta, t2, 0) - (analytic_a(alpha, beta, sigma, phi, eta, t1, 0)))
        - (analytic_b(alpha, t2, 0) - analytic_b(alpha, t1, 0)) * r_path[0]
        - (analytic_c(alpha, beta, t2, 0) - analytic_c(alpha, beta, t1, 0)) * theta_path[0]
    )
    for i in range(1, t2 * step):
        price_list[i] = \
            price_list[i-1] \
            + (analytic_b(alpha, t1, t[i]) - analytic_b(alpha, t2, t[i])) * price_list[i] * (r_path[i] - r_path[i-1]) \
            + (analytic_c(alpha, beta, t1, t[i]) - analytic_c(alpha, beta, t2, t[i])) * price_list[i] * (theta_path[i] - theta_path[i-1])
    return price_list[t1 * step] / price_list[t2 * step]


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


def forward_neutral_int_elur(T1, T2, r0, alpha, beta, sigma, theta0, phi, eta, t, w_sim_r, w_sim_theta):
    dt = t[1] - t[0]

    theta_path_for = np.zeros(len(t))
    theta_path_for[0] = theta0

    r_path_for = np.zeros(len(t))
    r_path_for[0] = r0

    # Euler theta and r
    for i in range(len(t) - 1):
        B_T1 = analytic_b(alpha, T1, t[i])
        B_T2 = analytic_b(alpha, T2, t[i])
        C_T1 = analytic_c(alpha, beta, T1, t[i])
        C_T2 = analytic_c(alpha, beta, T2, t[i])

        lambda1 = sigma * (B_T1 ** 2) / (B_T1 - B_T2)
        lambda2 = eta * (C_T1 ** 2) / (C_T1 - C_T2)

        theta_path_for[i + 1] = theta_path_for[i] + (beta * (phi - theta_path_for[i]) - lambda2 * eta) * dt + eta * (
                    w_sim_theta[i + 1] - w_sim_theta[i])
        r_path_for[i + 1] = r_path_for[i] + (
                    alpha * (theta_path_for[i] - r_path_for[i]) - lambda1 * sigma) * dt + sigma * (
                                        w_sim_r[i + 1] - w_sim_r[i])

    return theta_path_for, r_path_for


def forward_bond_price_T1(t1, t2, t, step, r_path, alpha, beta, sigma, eta, T1_Mat, T2_Mat):
    dt = t[1] - t[0]
    bank_account = np.zeros(int(t2 * step) + 1)
    bank_account[0] = 1
    for i in range(0, t2 * step):
        B_T1 = analytic_b(alpha, T1_Mat, t[i])
        B_T2 = analytic_b(alpha, T2_Mat, t[i])
        C_T1 = analytic_c(alpha, beta, T1_Mat, t[i])
        C_T2 = analytic_c(alpha, beta, T2_Mat, t[i])

        lambda1 = sigma * (B_T1 ** 2) / (B_T1 - B_T2)
        lambda2 = eta * (C_T1 ** 2) / (C_T1 - C_T2)

        bank_account[i + 1] = bank_account[i] + (r_path[i] + lambda1 * B_T1 * sigma + lambda2 * C_T1 * eta) * \
                              bank_account[i] * dt
        # bank_account[i+1] = bank_account[i]*np.exp(r_path[i]*dt)
    return bank_account[t1 * step] / bank_account[t2 * step]


def forward_bond_price_T2(t1, t2, t, step, r_path, alpha, beta, sigma, eta, T1_Mat, T2_Mat):
    dt = t[1] - t[0]
    bank_account = np.zeros(int(t2 * step) + 1)
    bank_account[0] = 1
    for i in range(0, t2 * step):
        B_T1 = analytic_b(alpha, T1_Mat, t[i])
        B_T2 = analytic_b(alpha, T2_Mat, t[i])
        C_T1 = analytic_c(alpha, beta, T1_Mat, t[i])
        C_T2 = analytic_c(alpha, beta, T2_Mat, t[i])

        lambda1 = sigma * (B_T1 ** 2) / (B_T1 - B_T2)
        lambda2 = eta * (C_T1 ** 2) / (C_T1 - C_T2)

        bank_account[i + 1] = bank_account[i] + (r_path[i] + lambda1 * B_T2 * sigma + lambda2 * C_T2 * eta) * \
                              bank_account[i] * dt
        # bank_account[i+1] = bank_account[i]*np.exp(r_path[i]*dt)
    return bank_account[t1 * step] / bank_account[t2 * step]


def forward_bank_account(t1, t2, t, step, r_path, alpha, beta, sigam, eta, T1_Mat, T2_Mat):
    dt = t[1] - t[0]
    bank_account = np.zeros(int(t2 * step) + 1)
    bank_account[0] = 1
    for i in range(0, t2 * step):
        B_T1 = analytic_b(alpha, T1_Mat, t[i])
        B_T2 = analytic_b(alpha, T2_Mat, t[i])
        C_T1 = analytic_c(alpha, beta, T1_Mat, t[i])
        C_T2 = analytic_c(alpha, beta, T2_Mat, t[i])

        lambda1 = sigma * (B_T1 ** 2) / (B_T1 - B_T2)
        lambda2 = eta * (C_T1 ** 2) / (C_T1 - C_T2)

        bank_account[i + 1] = bank_account[i] + (r_path[i] + lambda1 * B_T1 * sigma + lambda2 * C_T1 * eta) * \
                              bank_account[i] * dt
        # bank_account[i+1] = bank_account[i]*np.exp(r_path[i]*dt)
    return bank_account


def H1_sq(t1, t2, t, alpha, sigma):
    m1 = np.exp(-alpha * (t1 - t))
    m2 = np.exp(-alpha * (t2 - t))

    temp_H1 = sigma ** 2 / (2 * alpha ** 3) * ((np.exp(-alpha * (t2 - t1)) - 1) ** 2 - (m2 - m1) ** 2)
    return temp_H1


def H2_sq(t1, t2, t, alpha, beta, sigma, eta):
    m1 = np.exp(-alpha * (t1 - t))
    m2 = np.exp(-alpha * (t2 - t))
    n1 = np.exp(-beta * (t1 - t))
    n2 = np.exp(-beta * (t2 - t))

    p1 = beta ** 2 / (2 * alpha) * ((np.exp(-alpha * (t2 - t1)) - 1) ** 2 - (m2 - m1) ** 2)
    p2 = -2 * alpha * beta / (alpha + beta) * (
                (np.exp(-alpha * (t2 - t1)) - 1) * (np.exp(-beta * (t2 - t1)) - 1) - (m2 - m1) * (n2 - n1))
    p3 = alpha ** 2 / (2 * beta) * ((np.exp(-beta * (t2 - t1)) - 1) ** 2 - (n2 - n1) ** 2)

    H2_temp = eta ** 2 / (beta ** 2 * (alpha - beta) ** 2) * (p1 + p2 + p3)

    return H2_temp


def analytic_formula(r0, alpha, beta, sigma, theta0, phi, eta, T, t):
    return np.exp(- analytic_b(alpha, T, t) * r0
                  - analytic_c(alpha, beta, T, t) * theta0
                  + analytic_a(alpha, beta, sigma, phi, eta, T, t))


def bank_account(t1, t2, t, step, r_path):
    dt = t[1] - t[0]
    bank_account = np.zeros(int(t2 * step)+1)
    bank_account[0] = 1
    for i in range(0, t2 * step):
        bank_account[i+1] = bank_account[i] + r_path[i] * bank_account[i] * dt
        #bank_account[i+1] = bank_account[i]*np.exp(r_path[i]*dt)
    return bank_account


def swaption_sim(annuity, swap_rate_avg, nsims, tenure, tenure_steps, t, tsteps, int_matrix,t1, t2, k_param):
    # simulate swap rate at t0
    fixed_leg_sim = np.zeros(nsims)
    swap_rate_sim = np.zeros(nsims)
    value_sim = np.zeros(nsims)
    # vol = np.zeros(nsims)
    for n in range(0, nsims):
        for j in range(1, tenure_steps):
            fixed_leg_sim[n] += bond_price(t1, tenure[j], t, tsteps, int_matrix[n]) * 0.25
        swap_rate_sim[n] = (bond_price(t1, t1, t, tsteps, int_matrix[n])
                            - bond_price(t1, t2, t, tsteps, int_matrix[n])) / fixed_leg_sim[n]
        value_sim[n] = annuity * max(swap_rate_sim[n] - k_param * swap_rate_avg, 0)
        # print('vol' + str(n))
        # vol[n] = solve_imp_vol(value_sim[n], annuity, swap_rate_avg, k_param)

    # return np.mean(vol), annuity, swap_rate_avg
    return value_sim

def solve_imp_vol(v_0, a_0, s_0, alpha_k):
    imp_vol = np.arange(0.0001, 0.9999, 0.0001)
    price_diff = np.zeros_like(imp_vol)

    for i in range(len(imp_vol)):
        vol = imp_vol[i] * np.sqrt(3)
        # candidate = imp_vol[i] * np.sqrt(3)
        price_diff[i] = v_0 / (a_0 * s_0) - (
                (norm.cdf((np.log(1 / alpha_k) + 0.5 * vol ** 2) / vol))
                - alpha_k * (norm.cdf((np.log(1 / alpha_k) - 0.5 * vol ** 2) / vol))
        )
    idx = np.argmin(abs(price_diff))
    # print('vol_done')
    return imp_vol[idx]

def solve_imp_vol_zero(alpha_k):
    imp_vol = np.arange(0.00001, 0.99999, 0.00001)
    price_diff = np.zeros_like(imp_vol)

    for i in range(len(imp_vol)):
        vol = imp_vol[i] * np.sqrt(3)
        # candidate = imp_vol[i] * np.sqrt(3)
        price_diff[i] = alpha_k - (
                (norm.cdf((np.log(1 / alpha_k) + 0.5 * vol ** 2) / vol))
                / (norm.cdf((np.log(1 / alpha_k) - 0.5 * vol ** 2) / vol))
        )
    idx = np.argmin(abs(price_diff))
    # print('vol_done')
    return imp_vol[idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run different questions')
    parser.add_argument('--q2', action='store_true')
    parser.add_argument('--q3', action='store_true')
    parser.add_argument('--q4', action='store_true')
    parser.add_argument('--q5', action='store_true')

    args = parser.parse_args(['--q5'])

    if args.q2:

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

    if args.q3:
        # Q3
        theta_bond_yield = np.empty([5, nsteps])
        theta_range = [0.02, 0.05, 0.1, 0.3, 0.5]
        for i in range(5):
            for j in range(nsteps):
                theta_bond_yield[i, j] = analytic_formula_curve(r0, alpha, beta, sigma, theta_range[i], phi, eta, t[j], t=0)
        plt.figure(3)
        for i in range(5):
            plt.plot(t, theta_bond_yield[i])
        plt.xlabel('Time')
        plt.ylabel('Yield')
        plt.title('Analytic Term Structure of Yield Curve with Different Long Run Interest Rate')
        plt.legend(['\u0398_0 = 0.02', '\u0398_0 = 0.05', '\u0398_0 = 0.1', '\u0398_0 = 0.3', '\u0398_0 = 0.5'])
        plt.savefig('A3_Q3_theta.jpg')

        r_bond_yield = np.empty([5, nsteps])
        r_range = [0.02, 0.05, 0.1, 0.3, 0.5]
        for i in range(5):
            for j in range(nsteps):
                r_bond_yield[i, j] = analytic_formula_curve(r_range[i], alpha, beta, sigma, theta0, phi, eta, t[j], t=0)
        plt.figure(4)
        for i in range(5):
            plt.plot(t, r_bond_yield[i])
        plt.xlabel('Time')
        plt.ylabel('Yield')
        plt.title('Analytic Term Structure of Yield Curve with Different Short Run Interest Rate')
        plt.legend(['r_0 = 0.02', 'r_0 = 0.05', 'r_0 = 0.1', 'r_0 = 0.3', 'r_0 = 0.5'])
        plt.savefig('A3_Q3_r.jpg')

        alpha_bond_yield = np.empty([5, nsteps])
        alpha_range = [0.02, 0.1, 0.5, 0.8, 2]
        for i in range(5):
            for j in range(nsteps):
                alpha_bond_yield[i, j] = analytic_formula_curve(r0, alpha_range[i], beta, sigma, theta0, phi, eta, t[j], t=0)
        plt.figure(5)
        for i in range(5):
            plt.plot(t, alpha_bond_yield[i])
        plt.xlabel('Time')
        plt.ylabel('Yield')
        plt.title('Analytic Term Structure of Yield Curve with Different Alpha')
        plt.legend(['\u03B1 = 0.02', '\u03B1 = 0.1', '\u03B1 = 0.5', '\u03B1 = 0.8', '\u03B1 = 2'])
        plt.savefig('A3_Q3_alpha.jpg')

        beta_bond_yield = np.empty([5, nsteps])
        beta_range = [0.02, 0.1, 0.5, 0.8, 2]
        for i in range(5):
            for j in range(nsteps):
                beta_bond_yield[i, j] = analytic_formula_curve(r0, alpha, beta_range[i], sigma, theta0, phi, eta, t[j], t=0)
        plt.figure(6)
        for i in range(5):
            plt.plot(t, beta_bond_yield[i])
        plt.xlabel('Time')
        plt.ylabel('Yield')
        plt.title('Analytic Term Structure of Yield Curve with Different Beta')
        plt.legend(['\u03B2 = 0.02', '\u03B2 = 0.1', '\u03B2 = 0.5', '\u03B2 = 0.8', '\u03B2 = 2'])
        plt.savefig('A3_Q3_beta.jpg')

        eta_bond_yield = np.empty([5, nsteps])
        eta_range = [0.005, 0.05, 0.1, 0.15, 0.2]
        for i in range(5):
            for j in range(nsteps):
                eta_bond_yield[i, j] = analytic_formula_curve(r0, alpha, beta, sigma, theta0, phi, eta_range[i], t[j], t=0)
        plt.figure(7)
        for i in range(5):
            plt.plot(t, eta_bond_yield[i])
        plt.xlabel('Time')
        plt.ylabel('Yield')
        plt.title('Analytic Term Structure of Yield Curve with Different Eta')
        plt.legend(['\u03B7 = 0.005', '\u03B7 = 0.05', '\u03B7 = 0.1', '\u03B7 = 0.15', '\u03B7 = 0.2'])
        plt.savefig('A3_Q3_eta.jpg')

        sigma_bond_yield = np.empty([5, nsteps])
        sigma_range = [0.005, 0.05, 0.1, 0.15, 0.2]
        for i in range(5):
            for j in range(nsteps):
                sigma_bond_yield[i, j] = analytic_formula_curve(r0, alpha, beta, sigma_range[i], theta0, phi, eta, t[j], t=0)
        plt.figure(8)
        for i in range(5):
            plt.plot(t, sigma_bond_yield[i])
        plt.xlabel('Time')
        plt.ylabel('Yield')
        plt.title('Analytic Term Structure of Yield Curve with Different Sigma')
        plt.legend(['\u03C3 = 0.005', '\u03C3 = 0.05', '\u03C3 = 0.1', '\u03C3 = 0.15', '\u03C3 = 0.2'])
        plt.savefig('A3_Q3_sigma.jpg')

        phi_bond_yield = np.empty([5, nsteps])
        phi_range = [0.005, 0.02, 0.1, 0.5, 0.8]
        for i in range(5):
            for j in range(nsteps):
                phi_bond_yield[i, j] = analytic_formula_curve(r0, alpha, beta, sigma, theta0, phi_range[i], eta, t[j], t=0)
        plt.figure(9)
        for i in range(5):
            plt.plot(t, phi_bond_yield[i])
        plt.xlabel('Time')
        plt.ylabel('Yield')
        plt.title('Analytic Term Structure of Yield Curve with Different Phi')
        plt.legend(['\u03C6 = 0.005', '\u03C6 = 0.02', '\u03C6 = 0.1', '\u03C6 = 0.5', '\u03C6 = 0.8'])
        plt.savefig('A3_Q3_phi.jpg')

    if args.q4:
        # Q4
        # Analytic Version\
        T1 = 3
        T2 = 5
        nsteps = 40
        nsims = 1000
        r0 = 0.02
        alpha = 3
        sigma = 0.01
        theta0 = 0.03
        beta = 1
        phi = 0.05
        eta = 0.005

        P0_T1 = analytic_formula(r0, alpha, beta, sigma, theta0, phi, eta, T1, 0)
        P0_T2 = analytic_formula(r0, alpha, beta, sigma, theta0, phi, eta, T2, 0)
        K = P0_T2 / P0_T1

        H1_T1_sq = H1_sq(T1, T2, 0, alpha, sigma)
        H2_T1_sq = H2_sq(T1, T2, 0, alpha, beta, sigma, eta)

        K_steps = 11
        K_factor = np.linspace(0.95, 1.05, K_steps)
        g_T1 = np.zeros(K_steps)

        for i in range(0, K_steps):
            K_temp = K * K_factor[i]

            d1 = (np.log(P0_T2 / (K_temp * P0_T1)) + 0.5 * (H1_T1_sq + H2_T1_sq)) / (np.sqrt(H1_T1_sq + H2_T1_sq))
            d2 = (np.log(P0_T2 / (K_temp * P0_T1)) - 0.5 * (H1_T1_sq + H2_T1_sq)) / (np.sqrt(H1_T1_sq + H2_T1_sq))
            g_T1[i] = P0_T2 * norm.cdf(d1) - P0_T1 * K_temp * norm.cdf(d2)

        # Risk-Netrual MC simulation
        theta_path_set = np.zeros((nsims, nsteps))
        r_path_set = np.zeros((nsims, nsteps))
        t = np.linspace(0, T2, nsteps)
        steps = int(nsteps / T2)

        for i in range(0, nsims):
            w_sim_r = Sim_Brownian_Motion(t)
            w_sim_theta = Sim_Brownian_Motion(t)
            [theta_path_set[i:], r_path_set[i:]] = risk_neutral_int_elur(r0, alpha, beta, sigma, theta0, phi, eta, t,
                                                                         w_sim_r, w_sim_theta)
        int_matrix = r_path_set.reshape(nsims, nsteps)

        K_steps = 11
        K_factor = np.linspace(0.95, 1.05, K_steps)

        p0_t1_sim = np.zeros(nsims)
        p0_t2_sim = np.zeros(nsims)
        pt1_t2_sim = np.zeros(nsims)
        option_price = np.zeros([nsims, K_steps])
        K_sim = np.zeros(nsims)
        K_temp_sim = np.zeros(nsims)
        g_T_risk_sim = np.zeros(K_steps)
        K_final = np.zeros(K_steps)

        for i in range(0, K_steps):
            for n in range(0, nsims):
                p0_t1_sim[n] = bond_price(0, T1, t, steps, int_matrix[n])
                p0_t2_sim[n] = bond_price(0, T2, t, steps, int_matrix[n])
                pt1_t2_sim[n] = bond_price(T1, T2, t, steps, int_matrix[n])
                K_sim[n] = p0_t2_sim[n] / p0_t1_sim[n]
                # K_sim[n] = P0_T2/P0_T1
                K_temp_sim[n] = K_sim[n] * K_factor[i]
                discount = bank_account(0, T2, t, steps, int_matrix[n])
                option_price[n, i] = np.maximum((pt1_t2_sim[n] - K_temp_sim[n]), 0) / discount[T1 * steps]
            g_T_risk_sim[i] = np.mean(option_price[:, i])
            K_final[i] = np.mean(K_temp_sim)

        # MC simulation Forward-netural
        theta_path_set_for = np.zeros((nsims, nsteps))
        r_path_set_for = np.zeros((nsims, nsteps))
        t = np.linspace(0, T2, nsteps)
        steps = int(nsteps / T2)

        for i in range(0, nsims):
            w_sim_r = Sim_Brownian_Motion(t)
            w_sim_theta = Sim_Brownian_Motion(t)
            [theta_path_set_for[i:], r_path_set_for[i:]] = forward_neutral_int_elur(T1, T2, r0, alpha, beta, sigma,
                                                                                    theta0, phi, eta, t, w_sim_r,
                                                                                    w_sim_theta)
        int_matrix_for = r_path_set_for.reshape(nsims, nsteps)

        K_steps = 11
        K_factor = np.linspace(0.95, 1.05, K_steps)

        p0_t1_for_sim = np.zeros(nsims)
        p0_t2_for_sim = np.zeros(nsims)
        pt1_t2_for_sim = np.zeros(nsims)
        option_price_for = np.zeros([nsims, K_steps])
        K_for_sim = np.zeros(nsims)
        K_temp_for_sim = np.zeros(nsims)
        g_T_forward_sim = np.zeros(K_steps)
        K_final_for = np.zeros(K_steps)

        for i in range(0, K_steps):
            for n in range(0, nsims):
                p0_t1_for_sim[n] = forward_bond_price_T1(0, T1, t, steps, int_matrix_for[n], alpha, beta, sigma, eta, T1, T2)
                p0_t2_for_sim[n] = forward_bond_price_T2(0, T2, t, steps, int_matrix_for[n], alpha, beta, sigma, eta, T1, T2)
                pt1_t2_for_sim[n] = forward_bond_price_T2(T1, T2, t, steps, int_matrix_for[n], alpha, beta, sigma, eta, T1, T2)
                K_for_sim[n] = p0_t2_for_sim[n] / p0_t1_for_sim[n]
                # K_for_sim[n] = P0_T2/P0_T1
                K_temp_for_sim[n] = K_for_sim[n] * K_factor[i]
                discount_for = forward_bank_account(0, T2, t, steps, int_matrix_for[n], alpha, beta, sigma, eta, T1, T2)
                option_price_for[n, i] = np.maximum((pt1_t2_for_sim[n] - K_temp_for_sim[n]), 0) / discount_for[
                    T1 * steps]
            g_T_forward_sim[i] = np.mean(option_price_for[:, i])
            K_final_for[i] = np.mean(K_temp_for_sim)

        plt.plot(K * K_factor, g_T1, '-D', label='Analytic Option Price')
        plt.plot(K * K_factor, g_T_risk_sim, '-o', label='Risk-Netural Simulated Option Price')
        plt.plot(K * K_factor, g_T_forward_sim, '--', label='Forward-Netural Simulated Option Price')
        plt.xlabel('Strike')
        plt.ylabel('Option Price')
        plt.title('Option Price Simulation under Different Measures')
        plt.legend()


    if args.q5:
        t1 = 3
        t2 = 6
        nsteps = 25
        tenure_steps = 13
        nsims = 10000
        r0 = 0.02
        alpha = 3
        sigma = 0.01
        theta0 = 0.03
        beta = 1
        phi = 0.05
        eta = 0.005
        steps = int((nsteps-1) / t2)
        tsteps = 4
        strike_set = np.linspace(0.95, 1.2, 11)

        t = np.linspace(0, t2, nsteps)
        tenure = np.linspace(t1, t2, tenure_steps)

        # MC simulation
        theta_path_set = np.zeros((nsims, nsteps))
        r_path_set = np.zeros((nsims, nsteps))

        for i in range(0, nsims):
            w_sim_r = Sim_Brownian_Motion(t)
            w_sim_theta = Sim_Brownian_Motion(t)
            [theta_path_set[i:], r_path_set[i:]] = risk_neutral_int_elur(r0, alpha, beta, sigma, theta0, phi, eta, t, w_sim_r, w_sim_theta)
        int_matrix = r_path_set.reshape(nsims, nsteps)

        fixed_leg = np.zeros(nsims)
        swap_rate = np.zeros(nsims)
        discount = np.zeros(nsims)
        for i in range(nsims):
            for j in range(1, tenure_steps):
                fixed_leg[i] += bond_price(0, tenure[j], t, tsteps, int_matrix[i]) * 0.25
            swap_rate[i] = (bond_price(0, t1, t, tsteps, int_matrix[i])
                            - bond_price(0, t2, t, tsteps, int_matrix[i])) / fixed_leg[i]
            discount[i] = bond_price(0, t1, t, tsteps, int_matrix[i])
        annuity = np.mean(fixed_leg)
        swap_rate_avg = np.mean(swap_rate)

        v0 = np.zeros((len(strike_set), nsims))
        eta = np.zeros((len(strike_set), nsims))
        eta_avg = np.zeros(len(strike_set))
        v_avg = np.zeros(len(strike_set))
        # a0 = np.zeros(len(strike_set))
        # s0 = np.zeros(len(strike_set))

        # test = solve_imp_vol(0, a_0=annuity, s_0=swap_rate_avg, alpha_k=1)

        for i in range(len(strike_set)):
            myfunc = lambda value: solve_imp_vol(value, a_0=annuity, s_0=swap_rate_avg, alpha_k=strike_set[i]) if value != 0 else np.NaN
            vfunc = np.vectorize(myfunc)
            print('swap'+str(i))
            v0[i, :] = swaption_sim(annuity, swap_rate_avg, nsims, tenure, tenure_steps, t, tsteps, int_matrix, t1, t2, strike_set[i])
            eta[i, :] = vfunc(v0[i, :])
            eta_avg[i] = np.nanmean(eta[i, :])
            v0[v0 == 0] = np.NaN
            v_avg[i] = np.nanmean(v0[i, :])
            # eta[i, :] = solve_imp_vol(v0[i, :], annuity, swap_rate_avg, strike_set[i])

        # eta_avg = np.asarray(eta.mean(1)).reshape(-1)
        # v_avg = np.asarray(v0.mean(1)).reshape(-1)

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot([a*swap_rate_avg for a in strike_set], eta_avg, 'g-', label='Black Implied Volatility')
        ax2.plot([a*swap_rate_avg for a in strike_set], v_avg, 'b-',label='Swaption Price')

        ax1.set_xlabel('Strike Price')
        ax1.set_ylabel('Volatility', color='g')
        ax2.set_ylabel('Swaption Price', color='b')
        plt.title('Relationship Between Swaption price, Strike Price, and Black Implied Volatility')
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
        ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
        plt.savefig('Q5_avg.jpg')

        # plt.plot([a*s0 for a in strike_set], eta, label='Black Implied Volatility')
        # plt.xlabel('Strike Price')
        # plt.ylabel('Volatility')
        # plt.title('Relationship Between Strike Price and Black Implied Volatility')
        # # plt.legend()
        # plt.savefig('Q5.jpg')
        #
        # plt.plot(v0, eta, label='Swaption Price')
        # plt.xlabel('Swaption Price')
        # plt.ylabel('Volatility')
        # plt.title('Relationship Between Black Implied Volatility and Swaption Price')
        # # plt.legend()
        # plt.savefig('Q5_s.jpg')













