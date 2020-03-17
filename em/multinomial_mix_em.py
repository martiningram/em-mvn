import numpy as np
from scipy.special import logsumexp


def e_step(em_log_pi, em_log_rho, x):

    log_pi_numerator = em_log_pi + x @ em_log_rho.T
    log_pi_denom = logsumexp(log_pi_numerator, axis=1, keepdims=True)
    log_gamma = log_pi_numerator - log_pi_denom

    return log_gamma


def m_step(log_gamma, x):

    # Log rho update -- not happy about the exps but not sure how to avoid?
    numerator = np.einsum('nk,nm->km', np.exp(log_gamma), x)
    denominator = np.einsum('nk,nm->k', np.exp(log_gamma), x)
    new_log_rho = np.log(numerator) - np.log(denominator.reshape(-1, 1))

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
