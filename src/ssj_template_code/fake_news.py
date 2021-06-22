import numpy as np
import numba
from .standard_incomplete_markets import (example_calibration, backward_iteration,
                                          get_lottery, forward_iteration, steady_state,
                                          policy_impulse_response, impulse_response)


"""Step 1: get curlyYs and curlyDs from backward iteration"""

def terminal_shock_results(ss, T, i, shock=1, h=1E-4):
    """Get first-order individual policy response to perturbation
    'shock' to 'i' at all horizons 0, ... , T-1"""

    # make sure steady-state value is array, add leading dimension
    ss_val = np.array(ss[i])[np.newaxis, ...]
        
    # make path with ss repeated on leading dimension for T periods
    shocked_path = np.repeat(ss_val, T, axis=0)
        
    # now add h times shock to input at T-1
    shocked_path[T-1] += h * shock
    
    # now use this shocked path as input, get a and c
    _, a, c = policy_impulse_response(ss, T, **{i: shocked_path})
    
    # right now, a[T-1] is 0 periods to shock, and a[0] is T-1 periods, etc.
    # switch order so that a[0] is 0 periods to shock, etc.
    return a[::-1], c[::-1]


def one_period_ahead(ss, a):
    """If a[t] is (shocked) asset policy at time t, find distribution
    D[t+1], assuming D[t] is the steady-state distribution"""
 
    D = np.empty_like(a)
    for t in range(D.shape[0]):
        a_i, a_pi = get_lottery(a[t], ss['a_grid'])
        D[t] = forward_iteration(ss['D'], ss['Pi'], a_i, a_pi)
    return D


def get_curlyYs_curlyDs(ss, T, shocks, h=1E-4):
    """For each shock 'i' in shocks, find curlyY[o][i] and curlyD[i]
    at all horizons up to T"""

    # start with "ghost run": what happens without any shocks?
    a_noshock, c_noshock = terminal_shock_results(ss, T, 'r', 0, h)
    D_noshock = one_period_ahead(ss, a_noshock)
    
    # initialize dicts of curlyYs[o][i] and curlyDs[i]
    curlyYs = {'A': {}, 'C': {}}
    curlyDs = {}
    
    # iterate through shocks
    for ii, shock in shocks.items():
        # if tuple, split into name of input k and identifier i of shock
        if isinstance(ii, tuple):
            k, i = ii
        else:
            k, i = ii, ii
        
        a, c = terminal_shock_results(ss, T, k, shock, h)
        D = one_period_ahead(ss, a)
        
        # aggregate using steady-state distribution to get curlyYs
        curlyYs['A'][i] = np.sum(ss['D']*(a - a_noshock), axis=(1,2)) / h
        curlyYs['C'][i] = np.sum(ss['D']*(c - c_noshock), axis=(1,2)) / h
        
        curlyDs[i] = (D - D_noshock) / h
    
    return curlyYs, curlyDs


"""Step 2: get curlyEs from expectations iteration"""

@numba.njit
def expectations_iteration(X, Pi, a_i, a_pi):
    """If X is n_s*n_a array of values at each gridpoint tomorrow,
    what is the expected value at each gridpoint today?"""

    # first, take expectations over all s' using Markov matrix Pi
    X = Pi @ X
    
    # next, take expectations over a' using policy lottery
    expX = np.empty_like(X)
    for s in range(a_i.shape[0]):
        for a in range(a_i.shape[1]):
            # expected value today of policy lottery reflects:
            # pi(s,a) chance we go to gridpoint i(s,a)
            # 1-pi(s,a) chance we go to gridpoint i(s,a)+1
            expX[s, a] = (a_pi[s, a] * X[s, a_i[s, a]]
                          + (1 - a_pi[s, a]) * X[s, a_i[s, a]+1])
            
    return expX


def get_curlyEs(ss, T):
    """For outputs o = A and C, find expected value curlyE^o_u
    of o, u periods in the future, at each gridpoint today"""

    # initialize T*n_s*n_a arrays for curlyE^A and curlyE^C
    curlyEs = {'A': np.empty((T,) + ss['a'].shape),
               'C': np.empty((T,) + ss['c'].shape)}
    
    # at u=0, it's just the steady state policy
    curlyEs['A'][0] = ss['a']
    curlyEs['C'][0] = ss['c']
    
    # get steady-state policy lottery
    a_i, a_pi = get_lottery(ss['a'], ss['a_grid'])
    
    # recursively take expectations to get curlyE_u for u=1,...,T-1
    # (law of iterated expectations!)
    for o in ('A', 'C'):
        for u in range(1, T):
            curlyEs[o][u] = expectations_iteration(curlyEs[o][u-1],
                                                   ss['Pi'], a_i, a_pi)
            
    return curlyEs


"""Step 3: obtain fake news matrix from curlyY, curlyD, curlyE"""

@numba.njit
def fake_news(curlyY, curlyD, curlyE):
    T = len(curlyY)
    F = np.empty((T, T))

    # F[0, s] = curlyY(0, s)
    F[0, :] = curlyY

    # F[t, s] = dot product of curlyE(t-1) and curlyD(s) for t>0
    # (Numba only does vdot on 1-dim arrays, use ravel() to flatten to 1-dim)
    for t in range(1, T):
        for s in range(T):
            F[t, s] = np.vdot(curlyE[t-1].ravel(), curlyD[s].ravel())

    return F


"""Step 4: obtain Jacobian from fake news matrix"""

def J_from_F(F):
    J = F.copy()
    for t in range(1, F.shape[0]):
        J[1:, t] += J[:-1, t-1]
    return J


"""Combining everything"""


def jacobians(ss, T, shocks):
    curlyYs, curlyDs = get_curlyYs_curlyDs(ss, T, shocks)
    curlyEs = get_curlyEs(ss, T)
    
    Js = {'A': {}, 'C': {}}
    for o in Js:
        for i in curlyDs:
            F = fake_news(curlyYs[o][i], curlyDs[i], curlyEs[o])
            Js[o][i] = J_from_F(F)
            
    return Js