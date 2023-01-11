#from tqdm import tqdm
#import json
#import os
#import glob
#import pandas as pd
import numpy as np
from numba import jit

@jit(nopython=True)
def flatten_list(ll):
    N_out = 0
    for l0 in ll:
        N_out += len(l0)
    l_out = np.zeros(shape=N_out).astype(np.float64)
    counter = 0
    for l0 in ll:
        for l1 in l0:
            l_out[counter] = l1
            counter += 1
    return l_out

@jit(nopython=True)
def lognorm_pdf(x, mu, sigma):
    ''' PDF of the log-normal distribution.'''
    out = 1.0/(x*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(x)-mu)**2/(sigma**2))
    
    return out

@jit(nopython=True)
def norm_pdf(x, mu, sigma):
    ''' PDF of the normal distribution.'''
    out = 1.0/(sigma*np.sqrt(2*np.pi))*np.exp(-((x-mu)**2)/(sigma**2))

    return out

@jit(nopython=True)
def p_o_G_s(o, s, mu_l, sigma_l, sigma_alpha_2):
    ''' PDF of an observation *o* given a state *s*'''
    p_l = lognorm_pdf(o[0], mu_l[s], sigma_l[s]) 
    if s == 0:
        p_alpha = 1/(2*np.pi)
    else:
        p_alpha = norm_pdf(o[1], 0.0, sigma_alpha_2)

    p_o_G_s = p_l*p_alpha
    
    if p_o_G_s < 10**(-20):
        p_o_G_s = 10**(-20)
    
    return p_o_G_s

@jit(nopython=True)
def p_olog_G_s(o, s, mu_l, sigma_l, sigma_alpha_2):
    ''' PDF of the log of an observation *o* given a state *s*'''
    p_l = norm_pdf(o[0], mu_l[s], sigma_l[s]) 

    if s == 0:
        p_alpha = 1/(2*np.pi)
    else:
        p_alpha = norm_pdf(o[1], 0.0, sigma_alpha_2)

    p_o_G_s = p_l*p_alpha
    if p_o_G_s < 10**(-20):
        p_o_G_s = 10**(-20)
    return p_o_G_s

@jit(nopython=True)
def viterbi(T, pi, mu_l, sigma_l, sigma_alpha_2, O):
    """Viterbi algorithm for solving the uncovering problem

    adapted from:Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        T (np.ndarray): State transition probability matrix of dimension K x K
        pi (np.ndarray): Initial state distribution  of dimension K
        O (np.ndarray): Observation sequence of length N

    Returns:
        S_opt (np.ndarray): Optimal state sequence of length N
        D (np.ndarray): Accumulated probability matrix
        E (np.ndarray): Backtracking matrix
    """
    K = T.shape[0]    # Number of states
    N = len(O)  # Length of observation sequence

    # Initialize D and E matrices
    D = np.zeros((K, N)).astype(np.float64)
    E = np.zeros((K, N-1)).astype(np.int32)
    D[:, 0] = pi #np.multiply(C, B[:, O[0]])

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(K):
            temp_product = np.multiply(T[:, i], D[:, n-1])
            D[i, n] = np.max(temp_product) * p_o_G_s(O[n], i, mu_l, sigma_l, sigma_alpha_2)
            E[i, n-1] = np.argmax(temp_product)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D[:, -1])
    for n in range(N-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    return S_opt, D, E

@jit(nopython=True)
def Baum_Welch(O, T0, pi0, mu_l0, sigma_l0, sigma_alpha_2_0, fit_mu=True, fit_sigma=True, fit_pi=True, fit_T=True, fit_sigma_alpha=False):
    '''Baum-Welsh algoritm for solving the estimation problem
    
    '''
    O = O[np.isfinite(O[:,0])]
    K = T0.shape[0]    # Number of states
    N = len(O)  # Length of observation sequence
    
    alpha = np.zeros((K, N)).astype(np.float64)
    alpha_hat = np.zeros((K, N)).astype(np.float64)
    beta = np.ones((K, N)).astype(np.float64)
    pi_new = np.ones((K)).astype(np.float64)

    alpha[:, 0] = pi0*np.array([p_olog_G_s(O[0], 0, mu_l0, sigma_l0, sigma_alpha_2_0), p_olog_G_s(O[0], 1, mu_l0, sigma_l0, sigma_alpha_2_0)])
    alpha_hat_denom = np.sum(alpha[:, 0])
    for k in range(K):
        alpha_hat[k, 0] = alpha[k, 0]/alpha_hat_denom

    for n in range(1, N):
        for k in range(K):
            alpha_hat[k, n] = p_olog_G_s(O[n], k, mu_l0, sigma_l0, sigma_alpha_2_0)*np.sum(alpha_hat[:, n-1]*T0[k,:])
        alpha_hat_denom = np.sum(alpha_hat[:, n])
        for k in range(K):
            alpha_hat[k, n] = alpha_hat[k, n]/alpha_hat_denom

    for n in range(N-2, -1, -1):
        d_t = 0.0
        for k in range(K):
            beta[k, n] = np.sum(beta[:, n+1]*T0[k,:]*np.array([p_olog_G_s(O[n+1], 0, mu_l0, sigma_l0, sigma_alpha_2_0), p_olog_G_s(O[n+1], 1, mu_l0, sigma_l0, sigma_alpha_2_0)]))
            d_t = d_t + beta[k, n]
        d_t = 1.0/d_t
        beta[:,n] = d_t

    gamma_denom = np.sum(alpha_hat*beta, axis=0) 
    gamma = alpha_hat*beta
    for k in range(K):
        gamma[k] = gamma[k]/gamma_denom

    epsilon = np.zeros((K, K, N-1)).astype(np.float64)
    for n in range(N-1):
        for k in range(K):
            for k2 in range(K):
                epsilon[k, k2, n] = alpha_hat[k, n]*T0[k,k2]*beta[k2,n+1]*p_olog_G_s(O[n+1], k2, mu_l0, sigma_l0, sigma_alpha_2_0)
 
    epsilon_denom = np.zeros((N-1)).astype(np.float64)
    for k in range(K):
        for k2 in range(K):
             epsilon_denom += epsilon[k,k2]
    
    for k in range(K):
        for l in range(K):
            epsilon[k,l] = epsilon[k,l]/epsilon_denom
    
    for k in range(K):
        pi_new[k] = np.mean(gamma[k])
        
    mu_l_new = np.zeros(K).astype(np.float64)
    sigma_l_new = np.zeros(K).astype(np.float64)
    if fit_sigma_alpha:
        sigma_alpha2_new = np.sqrt(np.sum((O[:,1])**2*gamma[1])/np.sum(gamma[1]))
    else:
        sigma_alpha2_new = sigma_alpha_2_0
    T_new = np.zeros((K,K)).astype(np.float64)
    for k in range(K):
        mu_l_new[k] = np.sum(O[:,0]*gamma[k])/np.sum(gamma[k])
        sigma_l_new[k] = np.sqrt(np.sum((O[:,0]-mu_l0[k])**2*gamma[k])/np.sum(gamma[k]))
        T_denom = np.sum(gamma[k,:-1])
        for k2 in range(K):
            T_new[k,k2] = np.sum(epsilon[k,k2])/T_denom
    for k in range(K):
        T_new[k] = T_new[k]/np.sum(T_new[k])
    if np.isnan(T_new).any():
        T_new = 0.5*np.ones((2,2))
    sigma_l_new = sigma_l_new + 0.0000001
    
    if not fit_mu:
        mu_l_new = mu_l0
    if not fit_sigma:
        sigma_l_new = sigma_l0
    if not fit_pi:
        pi_new = pi0
    if not fit_T:
        T_new = T0
                  
    return mu_l_new, sigma_l_new, sigma_alpha2_new, T_new, pi_new

def convert_to_cartesian(o):
    ''' Convert observables o=[l,alpha] to [x,y] staring at [0,0]
    
    '''
    X = np.zeros(o.shape, dtype=float)
    X[:,1] = np.cumsum(o[:,1])
    X[:,0] = np.cumsum(o[:,0]*np.cos(X[:,1]))
    X[:,1] = np.cumsum(o[:,0]*np.sin(X[:,1]))
    
    return X

def convert_to_planar(X, dt=1.0):
    ''' Convert coordinates [x,y] to observables o=[l,alpha]
    
    '''
    X_diff = np.diff(X, axis=0)
    l = np.sqrt(X_diff[:,0]**2+X_diff[:,1]**2)/dt + np.random.uniform(0.0, 0.000001)
    correct_idx = (l>0.0).nonzero()
    gamma = np.arctan2(X_diff[:,1],X_diff[:,0])
    alpha = np.diff(gamma)
    alpha[alpha>np.pi] = alpha[alpha>np.pi] - 2*np.pi
    alpha[alpha<-np.pi] = alpha[alpha<-np.pi] + 2*np.pi

    
    return l[1:], alpha