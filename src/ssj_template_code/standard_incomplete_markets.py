"""
Simple code for standard incomplete markets model.
"""

import numpy as np
import numba


"""Part 0: example calibration from notebook"""

def example_calibration():
    y, _, Pi = discretize_income(0.975, 0.7, 7) 
    return dict(a_grid=discretize_assets(0, 10_000, 500),
                y=y, Pi=Pi, r=0.01/4, beta=1-0.08/4, eis=1., tau=0.)


"""Part 1: discretization tools"""

def discretize_assets(amin, amax, n_a):
    # find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = np.log(1 + np.log(1 + amax - amin))
    
    # make uniform grid
    u_grid = np.linspace(0, ubar, n_a)
    
    # double-exponentiate uniform grid and add amin to get grid from amin to amax
    return amin + np.exp(np.exp(u_grid) - 1) - 1


def rouwenhorst_Pi(N, p):
    # base case Pi_2
    Pi = np.array([[p, 1 - p],
                   [1 - p, p]])
    
    # recursion to build up from Pi_2 to Pi_N
    for n in range(3, N + 1):
        Pi_old = Pi
        Pi = np.zeros((n, n))
        
        Pi[:-1, :-1] += p * Pi_old
        Pi[:-1, 1:] += (1 - p) * Pi_old
        Pi[1:, :-1] += (1 - p) * Pi_old
        Pi[1:, 1:] += p * Pi_old
        Pi[1:-1, :] /= 2
        
    return Pi


def stationary_markov(Pi, tol=1E-14):
    # start with uniform distribution over all states
    n = Pi.shape[0]
    pi = np.full(n, 1/n)
    
    # update distribution using Pi until successive iterations differ by less than tol
    for _ in range(10_000):
        pi_new = Pi.T @ pi
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new


def discretize_income(rho, sigma, n_s):
    # choose inner-switching probability p to match persistence rho
    p = (1+rho)/2
    
    # start with states from 0 to n_s-1, scale by alpha to match standard deviation sigma
    s = np.arange(n_s)
    alpha = 2*sigma/np.sqrt(n_s-1)
    s = alpha*s
    
    # obtain Markov transition matrix Pi and its stationary distribution
    Pi = rouwenhorst_Pi(n_s, p)
    pi = stationary_markov(Pi)
    
    # s is log income, get income y and scale so that mean is 1
    y = np.exp(s)
    y /= np.vdot(pi, y)
    
    return y, pi, Pi


"""Part 2: Backward iteration for policy"""

def backward_iteration(Va, Pi, a_grid, y, r, beta, eis, tau):
    # step 1: discounting and expectations
    Wa = beta * Pi @ Va
    
    # step 2: solving for asset policy using the first-order condition
    c_endog = Wa**(-eis)
    coh = (1 - tau) * y[:, np.newaxis] + (1 + r) * a_grid
    
    a = np.empty_like(coh)
    for s in range(len(y)):
        a[s, :] = np.interp(coh[s, :], c_endog[s, :] + a_grid, a_grid)
        
    # step 3: enforcing the borrowing constraint and backing out consumption
    a = np.maximum(a, a_grid[0])
    c = coh - a
    
    # step 4: using the envelope condition to recover the derivative of the value function
    Va = (1+r) * c**(-1/eis)
    
    return Va, a, c


def policy_ss(Pi, a_grid, y, r, beta, eis, tau, tol=1E-9):
    # initial guess for Va: assume consumption 5% of cash-on-hand, then get Va from envelope condition
    coh = (1 - tau) * y[:, np.newaxis] + (1+r)*a_grid
    c = 0.05 * coh
    Va = (1 + r) * c**(-1/eis)
    
    # iterate until maximum distance between two iterations falls below tol, fail-safe max of 10,000 iterations
    for it in range(10_000):
        Va, a, c = backward_iteration(Va, Pi, a_grid, y, r, beta, eis, tau)
        
        # after iteration 0, can compare new policy function to old one
        if it > 0 and np.max(np.abs(a - a_old)) < tol:
            return Va, a, c
        
        a_old = a


"""Part 3: forward iteration for distribution"""

def get_lottery(a, a_grid):
    # step 1: find the i such that a' lies between gridpoints a_i and a_(i+1)
    a_i = np.maximum(np.searchsorted(a_grid, a) - 1, 0)
    
    # step 2: implement (8) to obtain lottery probabilities pi
    a_pi = (a_grid[a_i+1] - a)/(a_grid[a_i+1] - a_grid[a_i])
    
    return a_i, a_pi


@numba.njit
def forward_policy(D, a_i, a_pi):
    Dend = np.zeros_like(D)
    for s in range(a_i.shape[0]):
        for a in range(a_i.shape[1]):
            # send pi(s,a) of the mass to gridpoint i(s,a)
            Dend[s, a_i[s,a]] += a_pi[s,a]*D[s,a]
            
            # send 1-pi(s,a) of the mass to gridpoint i(s,a)+1
            Dend[s, a_i[s,a]+1] += (1-a_pi[s,a])*D[s,a]
            
    return Dend


def forward_iteration(D, Pi, a_i, a_pi):
    Dend = forward_policy(D, a_i, a_pi)    
    return Pi.T @ Dend


def distribution_ss(Pi, a, a_grid, tol=1E-10):
    a_i, a_pi = get_lottery(a, a_grid)
    
    # as initial D, use stationary distribution for s, plus uniform over a
    pi = stationary_markov(Pi)
    D = pi[:, np.newaxis] * np.ones_like(a_grid) / len(a_grid)
    
    # now iterate until convergence to acceptable threshold
    for _ in range(10_000):
        D_new = forward_iteration(D, Pi, a_i, a_pi)
        if np.max(np.abs(D_new - D)) < tol:
            return D_new
        D = D_new


"""Part 4: solving for steady state, including aggregates"""

def steady_state(Pi, a_grid, y, r, beta, eis, tau):
    Va, a, c = policy_ss(Pi, a_grid, y, r, beta, eis, tau)
    D = distribution_ss(Pi, a, a_grid)
    
    return dict(D=D, Va=Va, 
                a=a, c=c,
                A=np.vdot(a, D), C=np.vdot(c, D),
                Pi=Pi, a_grid=a_grid, y=y, r=r, beta=beta, eis=eis, tau=tau)


"""Part 5: dynamics: perfect-foresight impulse responses"""

def policy_impulse_response(ss, T, **shocks):
    assert all(x.shape[0] == T for x in shocks.values())
    
    # make list of all "current" inputs to the backward iteration function
    current_inputs = ['a_grid', 'y', 'r', 'beta', 'eis', 'tau']
    
    # make dict of all steady-state inputs that adds forward-looking Va and Pi
    # will use these for non-shocked inputs, and also Va and Pi in final period
    inputs = {k: ss[k] for k in current_inputs + ['Va', 'Pi']}
    
    # which of the current inputs are shocked?
    shocked_current_inputs = [k for k in current_inputs if k in shocks]
    
    # create a T*nS*nA dimensional array to store each outputs of backward iteration,
    # (Va, a, c), where nS*nA is the shape of each in steady state
    Va, a, c = (np.empty((T,) + ss['Va'].shape) for _ in range(3))
    
    for t in reversed(range(T)):
        # always use this period's value of any shocked current inputs
        for k in shocked_current_inputs:
            inputs[k] = shocks[k][t]
        
        # if not final period, use tomorrow's endogenous Va and possible shocked Pi
        if t < T-1:
            inputs['Va'] = Va[t+1]
            if 'Pi' in shocks:
                inputs['Pi'] = shocks['Pi'][t+1]
            
        Va[t], a[t], c[t] = backward_iteration(**inputs)
        
    return Va, a, c


def distribution_impulse_response(ss, T, **shocks):
    assert all(x.shape[0] == T for x in shocks.values())
    
    # create a T*nS*nA dimensional array to store distribution
    D = np.empty((T,) + ss['D'].shape)
    
    # at each t, we start by knowing D[t-1] and want D[t]
    # everything is assumed to be at steady state at -1 
    for t in range(T):        
        # if a is shocked, we want a[t-1], giving policy from t-1 to t
        a = shocks['a'][t-1] if ('a' in shocks and t > 0) else ss['a']

        # if a_grid is shocked, we want a_grid[t], giving grid at time t
        a_grid = shocks['a_grid'][t] if 'a_grid' in shocks else ss['a_grid']
        
        # use a from t-1 to t, and a_grid at t, to obtain lottery from t-1 to t
        a_i, a_pi = get_lottery(a, a_grid)
        
        # use Markov matrix Pi[t], which is for between t-1 and t, 
        # plus this lottery, to update D[t-1] to D[t]
        Pi = shocks['Pi'][t] if 'Pi' in shocks else ss['Pi']
        D_prev = D[t-1] if t > 0 else ss['D']
        
        D[t] = forward_iteration(D_prev, Pi, a_i, a_pi)
    
    return D


def impulse_response(ss, **shocks):
    # infer T from shocks, make sure all are the same
    shock_lengths = [x.shape[0] for x in shocks.values()]
    assert shock_lengths[1:] == shock_lengths[:-1]
    T = shock_lengths[0]
    
    # call backward and forward transitions
    Va, a, c = policy_impulse_response(ss, T, **shocks)
    D = distribution_impulse_response(ss, T, a=a, **shocks)
    
    # now get aggregate a and c: want the sum of D[t, ...]*a[t, ...] for each t
    A = np.sum(D*a, axis=(1, 2))
    C = np.sum(D*c, axis=(1, 2))
    
    return dict(Va=Va, a=a, c=c, D=D, A=A, C=C)
