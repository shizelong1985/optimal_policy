"""Computing asymptotic, discounted sums of output responses for use in steady state of the optimum calculations"""

import numpy as np

from sequence_jacobian.utilities import differentiate, interpolate, forward_step

# Most of the below code is borrowed and modified slightly from the HetBlock code, s.t. the optimal_policy repo only
# relies on SSJ utilities for now, as opposed to the entire code infrastructure.

# Need to be careful about setting `h`, since the initial `h`-scaled shock is not be computed in this current step,
# compared to the structure of the old get_curlyYs_curlyDs function.
# Hence, when calculating curlyYs and curlyDs w/ resp. to h, it needs to be the same `h` from which the initial `h`-scaled shock
# was calculated.
def backward_step_fakenews(ss, back_step_fun, inputs, outputs, back_iter_vars, back_iter_outputs, policy, shocked_inputs,
                           h=1e-4, recompute_policy_grid=True, ss_policy_repr=None, outputs_ss_vals=None):
    if ss_policy_repr is None:
        ss_policy_repr = get_sparse_ss_policy_repr(ss, policy)

    # shock perturbs outputs
    shocked_outputs = {k: v for k, v in zip(back_iter_outputs, differentiate.numerical_diff(back_step_fun,
                                                                                            {j: ss[j] for j in inputs},
                                                                                            shocked_inputs, h,
                                                                                            outputs_ss_vals))}
    curlyV = {k: shocked_outputs[k] for k in back_iter_vars}

    # which affects the distribution tomorrow
    # TODO: Generalize beyond 1d case. For the 1d case, `policy` should be a singleton list.
    pol = next(iter(policy))

    if recompute_policy_grid:
        # If the shock is large enough, it may cause the left grid point of the policy's sparse representation
        # to change. Hence, for robustness recompute the left grid points and weights of the shocked policy
        # to calculate the correct curlyD here.
        new_policy_repr = get_sparse_policy_repr({pol: ss[pol] + shocked_outputs[pol] * h}, {pol: ss[pol + "_grid"]})
        pol_i_old, pol_pi_old = ss_policy_repr[pol]["pol_i"], ss_policy_repr[pol]["pol_pi"]
        pol_i_new, pol_pi_new = new_policy_repr[pol]["pol_i"], new_policy_repr[pol]["pol_pi"]
        curlyD = (forward_step.forward_step_1d(ss["D"], ss["Pi"].T.copy(), pol_i_new, pol_pi_new) -
                  forward_step.forward_step_1d(ss["D"], ss["Pi"].T.copy(), pol_i_old, pol_pi_old)) / h

        # TODO: Figure out why this method of grid updating does not work...
        # new_policy_repr = get_sparse_policy_repr({pol: ss[pol] + shocked_outputs[pol] * h}, {pol: ss[pol + "_grid"]})
        # curlyD = forward_step.forward_step_shock_1d(ss["D"], ss["Pi"].T.copy(), new_policy_repr[pol]["pol_i"],
        #                                             -shocked_outputs[pol] / new_policy_repr[pol]["pol_space"])
    else:
        # This method will only work if we are sure that the grid points representing the policy's sparse
        # representation do not change. Should be feasible with small enough h, but then may run into
        # numerical differentiation precision issues.
        curlyD = forward_step.forward_step_shock_1d(ss["D"], ss["Pi"].T.copy(), ss_policy_repr[pol]["pol_i"],
                                                    -shocked_outputs[pol] / ss_policy_repr[pol]["pol_space"])

    # and the aggregate outcomes today
    curlyY = {k: np.vdot(ss["D"], shocked_outputs[k.lower()]) for k in outputs}

    return curlyV, curlyD, curlyY


def get_sparse_ss_policy_repr(ss, policy):
    policies = {k: ss[k] for k in policy}
    grids = {k: ss[k + '_grid'] for k in policy}
    return get_sparse_policy_repr(policies, grids)


def get_sparse_policy_repr(policies, grids):
    """Find the sparse representation of a given policy function f(x) on a grid of x[i], x[i+1], ... by the following
    formula: f(x) = w x[i] + (1 - w) x[i+1], where w in [0, 1].

    Inputs
    ------
    policies: `dict`
        A dict of the names of policies and the policy functions, defined over their relevant grids.
    grids: `dict`
        A dict of the names of policies and the grids they are defined over.

    Outputs
    -------
    policy_reprs: `dict`
        A dict of dicts, mapping names of policies to a dict with three entries:
        1) pol_i, the left grid points `i` in x[i] from the interpolation formula above
        2) pol_pi, the weights on the left grid points, `w`, from the the interpolation formula above
        3) pol_space, the distance between the adjacent gridpoints, x[i+1] - x[i], from the interpolation formula above
    """
    policy_reprs = {}

    # Get sparse representations of policy rules, and distance between neighboring policy gridpoints
    for pol in policies:
        policy_reprs[pol] = {}
        policy_reprs[pol]["pol_i"], policy_reprs[pol]["pol_pi"], policy_reprs[pol]["pol_space"] = {}, {}, {}
        # use robust binary-search-based method that only requires grids to be monotonic
        policy_reprs[pol]["pol_i"], policy_reprs[pol]["pol_pi"] = interpolate.interpolate_coord_robust(grids[pol], policies[pol])
        policy_reprs[pol]["pol_space"] = grids[pol][policy_reprs[pol]["pol_i"] + 1] - grids[pol][policy_reprs[pol]["pol_i"]]
    return policy_reprs
