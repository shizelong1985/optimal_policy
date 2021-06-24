"""Computing asymptotic, discounted sums of output responses for use in steady state of the optimum calculations"""

import numpy as np

import sequence_jacobian.utilities as utils
import src.ssj_template_code.standard_incomplete_markets as sim

# Most of the below code is borrowed and modified slightly from the HetBlock code, s.t. the optimal_policy repo only
# relies on SSJ utilities for now, as opposed to the entire code infrastructure.


def get_sparse_policy_repr(ss, policy):
    grid = {k: ss[k + '_grid'] for k in policy}
    # Get sparse representations of policy rules, and distance between neighboring policy gridpoints
    sspol_i = {}
    sspol_pi = {}
    sspol_space = {}
    for pol in policy:
        # use robust binary-search-based method that only requires grids to be monotonic
        sspol_i[pol], sspol_pi[pol] = utils.interpolate.interpolate_coord_robust(grid[pol], ss[pol])
        sspol_space[pol] = grid[pol][sspol_i[pol]+1] - grid[pol][sspol_i[pol]]
    return {"pol_i": sspol_i, "pol_pi": sspol_pi, "pol_space": sspol_space}


# Need to be careful about setting `h`, since the initial `h`-scaled shock is not be computed in this current step,
# compared to the structure of the old get_curlyYs_curlyDs function.
# Hence, when calculating curlyYs and curlyDs w/ resp. to h, it needs to be the same `h` from which the initial `h`-scaled shock
# was calculated.
def backward_step_fakenews(ss, inputs, outputs, back_iter_vars, back_iter_outputs, policy, shocked_inputs, h=1e-4,
                           policy_repr=None, outputs_ss_vals=None):
    if policy_repr is None:
        policy_repr = get_sparse_policy_repr(ss, policy)

    # shock perturbs outputs
    shocked_outputs = {k: v for k, v in zip(back_iter_outputs, utils.differentiate.numerical_diff(sim.backward_iteration,
                                                                                        {j: ss[j] for j in inputs},
                                                                                        shocked_inputs, h,
                                                                                        outputs_ss_vals))}
    curlyV = {k: shocked_outputs[k] for k in back_iter_vars}

    # which affects the distribution tomorrow
    p = next(iter(policy))  # For the 1d case, `policy` should be a singleton list
    pol_pi_shock = -shocked_outputs[p]/policy_repr["pol_space"][p]
    # TODO: Test, manually construct pol_pi_shock
    #   pol_pi_shock matches pol_pi_shock_manual, except for a few entries that seem to be due to numerical
    #   differentiation error (the discrepancy is of order 1/h in these entries)
    a_i_old, a_pi_old = sim.get_lottery(ss["a"], ss["a_grid"])
    # a_i_new, a_pi_new = sim.get_lottery(ss["a"] + shocked_outputs["a"] * h, ss["a_grid"])
    # pol_pi_shock_manual = (a_pi_new - a_pi_old)/h
    a_i_new, a_pi_new = sim.get_lottery(ss["a"] + shocked_outputs["a"] * h, ss["a_grid"])
    pol_pi_shock_manual = (a_pi_new - a_pi_old)/h

    curlyD = utils.forward_step.forward_step_shock_1d(ss["D"], ss["Pi"].T.copy(), policy_repr["pol_i"][p],
                                                      # -shocked_outputs[p]/policy_repr["pol_space"][p])
                                                      pol_pi_shock)
    # TODO: a_i_new \neq a_i_old!! So might be some weirdness going on when using pol_pi_shock!
    # # Matches the curlyD calculation in SIM
    # curlyD = (utils.forward_step.forward_step_1d(ss["D"], ss["Pi"].T.copy(), a_i_new, a_pi_new) -\
    #           utils.forward_step.forward_step_1d(ss["D"], ss["Pi"].T.copy(), a_i_old, a_pi_old)) / h
    # Matches forward_step in SIM (doing D instead of (D - D_noshock)/h)
    # curlyD = utils.forward_step.forward_step_1d(ss["D"], ss["Pi"].T.copy(), a_i_new, a_pi_new)

    # and the aggregate outcomes today
    curlyY = {k: np.vdot(ss["D"], shocked_outputs[k.lower()]) for k in outputs}

    return curlyV, curlyD, curlyY
