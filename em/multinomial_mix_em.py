import numpy as np
from scipy.special import logsumexp


def e_step(em_log_pi, em_log_rho, x):

    log_pi_numerator = em_log_pi + x @ em_log_rho.T
    log_pi_denom = logsumexp(log_pi_numerator, axis=1, keepdims=True)
    log_gamma = log_pi_numerator - log_pi_denom

    return log_gamma


def safe_log_rho_numerator(log_gamma, x):

    K = log_gamma.shape[1]
    M = x.shape[1]

    log_result = np.zeros((K, M))

    # Gamma is N x K
    # x is N x M
    for m in range(x.shape[1]):

        cur_x = x[:, m]
        to_keep = cur_x > 0
        x_to_sum = cur_x[to_keep]

        log_x = np.log(x_to_sum)
        cur_log_gamma = log_gamma[to_keep]
        # Log x will be N*
        # Gamma will be N* x K

        # Sum them up
        cur_res = log_x.reshape(-1, 1) + cur_log_gamma
        summed = logsumexp(cur_res, axis=0)

        log_result[:, m] = summed

    return log_result


def m_step(log_gamma, x):

    # Log rho update -- not happy about the exps but not sure how to avoid?
    # numerator = np.einsum('nk,nm->km', np.exp(log_gamma), x)
    log_numerator = safe_log_rho_numerator(log_gamma, x)
    # log_numerator = np.log(numerator)

    # This will be N x 1
    a = np.sum(x, axis=1, keepdims=True)
    summed_denom = np.log(a) + log_gamma
    denominator = logsumexp(summed_denom, axis=0)

    # denominator = np.einsum('nk,nm->k', np.exp(log_gamma), x)
    new_log_rho = log_numerator - denominator.reshape(-1, 1)

    # Log pi update
    N = x.shape[0]
    new_log_pi = -np.log(N) + logsumexp(log_gamma, axis=0)

    return new_log_rho, new_log_pi


def log_likelihood(em_log_pi, em_log_rho, x):

    inner_part = em_log_pi + x @ em_log_rho.T
    log_inner_part = logsumexp(inner_part, axis=1)

    return np.sum(log_inner_part)


def fit_em(init_log_pi, init_log_rho, x, tol=1e-8, maxiter=int(1E6)):

    em_log_pi = init_log_pi
    em_log_rho = init_log_rho

    prev_log_lik = -1
    cur_log_lik = 0
    iteration = 0

    liks = list()

    while (iteration <= 1
           or np.abs(cur_log_lik - prev_log_lik) > tol
           and iteration < maxiter):

        prev_log_lik = cur_log_lik

        log_gamma = e_step(em_log_pi, em_log_rho, x)
        em_log_rho, em_log_pi = m_step(log_gamma, x)
        cur_log_lik = log_likelihood(em_log_pi, em_log_rho, x)

        liks.append(cur_log_lik)

        iteration += 1

    liks = np.array(liks)

    return em_log_rho, em_log_pi, liks
