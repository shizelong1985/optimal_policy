"""Computing asymptotic, discounted sums of output responses for use in steady state of the optimum calculations"""

import numpy as np

import src.ssj_template_code.standard_incomplete_markets as sim


# Re-write policy_impulse_response to be a single backward iteration step
# Need to be careful about setting `h`, since the initial `h`-scaled shock is not be computed in this current step,
# compared to the structure of the old get_curlyYs_curlyDs function.
# Hence, when calculating curlyYs and curlyDs w/ resp. to h, it needs to be the same `h` from which the initial `h`-scaled shock
# was calculated.
def get_curlyYs_curlyDs_single_step(ss, shocked_inputs,
                                    inputs=['a_grid', 'y', 'r', 'beta', 'eis', 'Va', 'Pi'], h=1e-4):
    input_args_noshock = {k: ss[k] for k in inputs}
    input_args = input_args_noshock.copy()

    for k in [j for j in shocked_inputs if j in inputs]:
        input_args[k] = shocked_inputs[k]

    # initialize dicts of curlyYs[o][i] and curlyDs[i]
    curlyYs = {}
    for o in ['A', 'C']:  # TODO: Later generalize beyond the simple SIM case...
        curlyYs[o] = {}

    # start with "ghost run": what happens without any shocks?
    _, a_noshock, c_noshock = sim.backward_iteration(**input_args_noshock)
    D_noshock = one_period_ahead_single_step(ss, a_noshock)

    # compute the actual policy response to the shocks
    _, a, c = sim.backward_iteration(**input_args)
    D = one_period_ahead_single_step(ss, a)

    # aggregate using steady-state distribution to get curlyYs
    curlyYs['A'] = np.vdot(ss['D'], (a - a_noshock)) / h
    curlyYs['C'] = np.vdot(ss['D'], (c - c_noshock)) / h

    curlyDs = (D - D_noshock) / h

    return curlyYs, curlyDs


def one_period_ahead_single_step(ss, a):
    """If a[t] is (shocked) asset policy at time t, find distribution
    D[t+1], assuming D[t] is the steady-state distribution"""

    a_i, a_pi = sim.get_lottery(a, ss['a_grid'])
    D = sim.forward_iteration(ss['D'], ss['Pi'], a_i, a_pi)
    return D
