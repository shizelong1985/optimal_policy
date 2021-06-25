"""Computing asymptotic, discounted sums of output responses for use in steady state of the optimum calculations"""

import copy
import numpy as np

from sequence_jacobian.utilities import differentiate, interpolate, forward_step
import src.ssj_template_code.fake_news as fn
from src.utils import make_inputs


def asymp_disc_sums(ss, back_step_fun, inputs, outputs, back_iter_vars, back_iter_outputs, policy, exogenous,
                    shocked_inputs, h=1e-4, demean_curlyEs=True, recompute_policy_grid=True, ss_policy_repr=None,
                    outputs_ss_vals=None, return_Ts=False, verbose=False, maxit=1000, tol=1e-8):
    curlyDs_sum, curlyYs_sum, T_for_Ds_and_Ys = asymp_disc_sums_curlyDs_and_Ys(ss, back_step_fun, inputs, outputs,
                                                                               back_iter_vars, back_iter_outputs,
                                                                               policy, exogenous, shocked_inputs, h=h,
                                                                               recompute_policy_grid=recompute_policy_grid,
                                                                               ss_policy_repr=ss_policy_repr,
                                                                               outputs_ss_vals=outputs_ss_vals,
                                                                               verbose=verbose, maxit=maxit, tol=tol)
    curlyEs_sum, T_for_Es = asymp_disc_sum_curlyE(ss, outputs, policy, demean=demean_curlyEs,
                                                  ss_policy_repr=ss_policy_repr, verbose=verbose,
                                                  maxit=maxit, tol=tol)
    asymp_disc_sums = {}
    for o in outputs:
        asymp_disc_sums[o] = {}
        for s in shocked_inputs.keys():
            asymp_disc_sums[o][s] = curlyYs_sum[o][s] + ss["beta"] * np.vdot(curlyEs_sum[o], curlyDs_sum[s])

    if return_Ts:
        return asymp_disc_sums, (T_for_Ds_and_Ys, T_for_Es)
    else:
        return asymp_disc_sums


def asymp_disc_sums_curlyDs_and_Ys(ss, back_step_fun, inputs, outputs, back_iter_vars, back_iter_outputs, policy,
                                   exogenous, shocked_inputs, h=1e-4, recompute_policy_grid=True, ss_policy_repr=None,
                                   outputs_ss_vals=None, verbose=False, checkit=10, maxit=1000, tol=1e-8):
    for i in range(maxit):
        if i == 0:
            curlyVs, curlyDs, curlyYs = backward_step_fakenews(ss, back_step_fun, inputs, outputs,
                                                               back_iter_vars, back_iter_outputs,
                                                               policy, exogenous, shocked_inputs, h=h,
                                                               recompute_policy_grid=recompute_policy_grid,
                                                               ss_policy_repr=ss_policy_repr,
                                                               outputs_ss_vals=outputs_ss_vals)
            curlyDs_sum, curlyYs_sum = copy.deepcopy(curlyDs), copy.deepcopy(curlyYs)
        else:
            for shock_name in shocked_inputs.keys():
                ss_new = ss.copy()
                for back_iter_var in back_iter_vars:
                    ss_new[back_iter_var] = ss[back_iter_var] + curlyVs[back_iter_var][shock_name] * h
                curlyVs_aux, curlyDs_aux, curlyYs_aux = backward_step_fakenews(ss_new, back_step_fun, inputs, outputs,
                                                                               back_iter_vars, back_iter_outputs,
                                                                               policy, exogenous, {shock_name: 0.}, h=h,
                                                                               recompute_policy_grid=recompute_policy_grid,
                                                                               ss_policy_repr=ss_policy_repr,
                                                                               outputs_ss_vals=outputs_ss_vals)
                for back_iter_var in back_iter_vars:
                    curlyVs[back_iter_var][shock_name] = curlyVs_aux[back_iter_var][shock_name]
                curlyDs[shock_name] = curlyDs_aux[shock_name]
                # TODO: An optimization later could be not calculating curlyYs or curlyDs if either converges earlier
                curlyDs_sum[shock_name] += ss["beta"] ** -i * curlyDs[shock_name]
                for o in outputs:
                    curlyYs[o][shock_name] = curlyYs_aux[o][shock_name]
                    curlyYs_sum[o][shock_name] += ss["beta"] ** -i * curlyYs[o][shock_name]

        curlyD_max_abs_diff = max([np.max(np.abs(ss["beta"] ** -i * curlyDs[s])) for s in shocked_inputs])
        curlyY_max_abs_diff = max([max([np.max(np.abs(ss["beta"] ** -i * curlyYs[o][s])) for o in outputs]) for s in shocked_inputs])

        if i % checkit == 0 and verbose:
            print(f"Iteration {i} max abs change in curlyD sum is {curlyD_max_abs_diff} and in curlyY sum is {curlyY_max_abs_diff}")
        if i % checkit == 0 and curlyD_max_abs_diff < tol and curlyY_max_abs_diff < tol:
            return curlyDs_sum, curlyYs_sum, i
        if i == maxit - 1:
            raise ValueError(f'No convergence of asymptotic discounted sums after {maxit} iterations!')


def asymp_disc_sum_curlyE(ss, outputs, policy, demean=True, ss_policy_repr=None, verbose=False,
                          checkit=10, maxit=1000, tol=1e-8):
    if ss_policy_repr is None:
        ss_policy_repr = get_sparse_ss_policy_repr(ss, policy)
    if demean:
        output_means = {o: np.vdot(ss["D"], ss[o.lower()]) for o in outputs}

    pol = next(iter(policy))  # TODO: Generalize beyond 1d case. For the 1d case, `policy` should be a singleton list.
    for i in range(maxit):
        if i == 0:
            curlyEs = {o: ss[o.lower()] for o in outputs}
            curlyEs_sum = {o: curlyEs[o] - output_means[o] for o in outputs} if demean else curlyEs
        else:
            for o in outputs:
                curlyEs[o] = fn.expectations_iteration(curlyEs[o], ss["Pi"], ss_policy_repr[pol]["pol_i"],
                                                       ss_policy_repr[pol]["pol_pi"])
                curlyEs_sum[o] += ss["beta"] ** i * (curlyEs[o] - output_means[o]) if demean else ss["beta"] ** i * curlyEs[o]

        curlyE_abs_diff = {}
        for o in outputs:
            curlyE_abs_diff[o] = np.abs(ss["beta"] ** i * (curlyEs[o] - output_means[o]) if demean else ss["beta"] ** i * curlyEs[o])
        curlyE_max_abs_diff = max([np.max(curlyE_abs_diff[o]) for o in outputs])
        if i % checkit == 0 and verbose:
            print(f"Iteration {i} max abs change in curlyE sum is {curlyE_max_abs_diff}")
        if i % checkit == 0 and curlyE_max_abs_diff < tol:
            return curlyEs_sum, i
        if i == maxit - 1:
            raise ValueError(f'No convergence of asymptotic discounted sums after {maxit} iterations!')


# Need to be careful about setting `h`, since the initial `h`-scaled shock is not be computed in this current step,
# compared to the structure of the old get_curlyYs_curlyDs function.
# Hence, when calculating curlyYs and curlyDs w/ resp. to h, it needs to be the same `h` from which the initial `h`-scaled shock
# was calculated.
def backward_step_fakenews(ss, back_step_fun, inputs, outputs, back_iter_vars, back_iter_outputs, policy,
                           exogenous, shocked_inputs, h=1e-4, recompute_policy_grid=True,
                           ss_policy_repr=None, outputs_ss_vals=None):
    if ss_policy_repr is None:
        ss_policy_repr = get_sparse_ss_policy_repr(ss, policy)
    if outputs_ss_vals is None:
        outputs_ss_vals = tuple(ss[i] for i in back_iter_outputs)

    # Initialize variables
    Pi_T = ss["Pi"].T.copy()
    shocked_outputs = {b: {} for b in back_iter_outputs}
    curlyVs = {b: {} for b in back_iter_vars}
    curlyYs = {o: {} for o in outputs}
    curlyDs = {s: {} for s in shocked_inputs.keys()}

    # TODO: Generalize beyond 1d case. For the 1d case, `policy` should be a singleton list.
    pol = next(iter(policy))

    # 1) Shock perturbs outputs
    for shock_name, shock_value in shocked_inputs.items():
        inputs_dict = make_inputs(inputs, ss, back_iter_vars, exogenous)
        out = differentiate.numerical_diff(back_step_fun, inputs_dict, {shock_name: shock_value}, h, outputs_ss_vals)
        for ib, b in enumerate(back_iter_outputs):
            shocked_outputs[b][shock_name] = out[ib]
            if b in back_iter_vars:
                curlyVs[b][shock_name] = shocked_outputs[b][shock_name]

        # 2) which affects the distribution tomorrow
        if recompute_policy_grid:
            # If the shock is large enough, it may cause the left grid point of the policy's sparse representation
            # to change. Hence, for robustness recompute the left grid points and weights of the shocked policy
            # to calculate the correct curlyD here.
            new_policy_repr = get_sparse_policy_repr({pol: ss[pol] + shocked_outputs[pol][shock_name] * h},
                                                     {pol: ss[pol + "_grid"]})
            pol_i_old, pol_pi_old = ss_policy_repr[pol]["pol_i"], ss_policy_repr[pol]["pol_pi"]
            pol_i_new, pol_pi_new = new_policy_repr[pol]["pol_i"], new_policy_repr[pol]["pol_pi"]
            curlyDs[shock_name] = (forward_step.forward_step_1d(ss["D"], Pi_T, pol_i_new, pol_pi_new) -
                                   forward_step.forward_step_1d(ss["D"], Pi_T, pol_i_old, pol_pi_old)) / h

            # TODO: Figure out why this method of grid updating does not work...
            # new_policy_repr = get_sparse_policy_repr({pol: ss[pol] + shocked_outputs[pol][shock_name] * h},
            #                                          {pol: ss[pol + "_grid"]})
            # curlyDs[shock_name] = forward_step.forward_step_shock_1d(ss["D"], Pi_T, new_policy_repr[pol]["pol_i"],
            #                                                          -shocked_outputs[pol][shock_name] /
            #                                                          new_policy_repr[pol]["pol_space"])
        else:
            # This method will only work if we are sure that the grid points representing the policy's sparse
            # representation do not change. Should be feasible with small enough h, but then may run into
            # numerical differentiation precision issues.
            curlyDs[shock_name] = forward_step.forward_step_shock_1d(ss["D"], Pi_T, ss_policy_repr[pol]["pol_i"],
                                                                     -shocked_outputs[pol][shock_name] /
                                                                     ss_policy_repr[pol]["pol_space"])
        # 3) and the aggregate outcomes today
        for o in outputs:
            curlyYs[o][shock_name] = np.vdot(ss["D"], shocked_outputs[o.lower()][shock_name])

    return curlyVs, curlyDs, curlyYs


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
