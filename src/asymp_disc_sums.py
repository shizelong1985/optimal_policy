"""Computing asymptotic, discounted sums of output responses for use in steady state of the optimum calculations"""

import numpy as np

from sequence_jacobian.utilities import differentiate, interpolate, forward_step
import src.ssj_template_code.fake_news as fn


def asymp_disc_sums(ss, back_step_fun, inputs, outputs, back_iter_vars, back_iter_outputs, policy, shocked_inputs,
                    h=1e-4, demean_curlyEs=True, recompute_policy_grid=True, ss_policy_repr=None, outputs_ss_vals=None,
                    return_Ts=False, verbose=False, maxit=1000, tol=1e-8):
    curlyDs_sum, curlyYs_sum, T_for_Ds_and_Ys = asymp_disc_sums_curlyDs_and_Ys(ss, back_step_fun, inputs, outputs,
                                                                               back_iter_vars, back_iter_outputs,
                                                                               policy, shocked_inputs, h=h,
                                                                               recompute_policy_grid=recompute_policy_grid,
                                                                               ss_policy_repr=ss_policy_repr,
                                                                               outputs_ss_vals=outputs_ss_vals,
                                                                               verbose=verbose, maxit=maxit, tol=tol)
    curlyEs_sum, T_for_Es = asymp_disc_sum_curlyE(ss, outputs, policy, demean=demean_curlyEs,
                                                  ss_policy_repr=ss_policy_repr, verbose=verbose,
                                                  maxit=maxit, tol=tol)
    asymp_disc_sums = {}
    for o in outputs:
        asymp_disc_sums[o] = curlyYs_sum[o] + ss["beta"] * np.vdot(curlyEs_sum[o], curlyDs_sum)

    if return_Ts:
        return asymp_disc_sums, (T_for_Ds_and_Ys, T_for_Es)
    else:
        return asymp_disc_sums


def asymp_disc_sums_curlyDs_and_Ys(ss, back_step_fun, inputs, outputs, back_iter_vars, back_iter_outputs, policy, shocked_inputs,
                                   h=1e-4, recompute_policy_grid=True, ss_policy_repr=None, outputs_ss_vals=None,
                                   verbose=False, maxit=1000, tol=1e-8):
    for i in range(maxit):
        if i == 0:
            curlyVs, curlyDs, curlyYs = backward_step_fakenews(ss, back_step_fun, inputs, outputs,
                                                               back_iter_vars, back_iter_outputs,
                                                               policy, shocked_inputs, h=h,
                                                               recompute_policy_grid=recompute_policy_grid,
                                                               ss_policy_repr=ss_policy_repr,
                                                               outputs_ss_vals=outputs_ss_vals)
            curlyDs_sum, curlyYs_sum = curlyDs, curlyYs
        else:
            ss_new = ss.copy()
            ss_new["Va"] = ss["Va"] + curlyVs["Va"] * h
            curlyVs, curlyDs, curlyYs = backward_step_fakenews(ss_new, back_step_fun, inputs, outputs,
                                                               back_iter_vars, back_iter_outputs,
                                                               policy, {}, h=h,
                                                               recompute_policy_grid=recompute_policy_grid,
                                                               outputs_ss_vals=outputs_ss_vals)
            # TODO: An optimization later could be not calculating curlyYs or curlyDs if either converges earlier
            curlyDs_sum += ss["beta"] ** -i * curlyDs
            for o in outputs:
                curlyYs_sum[o] += ss["beta"] ** -i * curlyYs[o]

        # TODO: Consolidate this post-debugging
        curlyD_abs_diff = np.abs(ss["beta"] ** -i * curlyDs)
        curlyY_abs_diff = {}
        for o in outputs:
            curlyY_abs_diff[o] = np.abs(ss["beta"] ** -i * curlyYs[o])
        curlyD_max_abs_diff = np.max(curlyD_abs_diff)
        curlyY_max_abs_diff = max(curlyY_abs_diff.values())

        if i % 10 == 1 and verbose:
            print(f"Iteration {i} max abs change in curlyD sum is {curlyD_max_abs_diff} and in curlyY sum is {curlyY_max_abs_diff}")
        if i % 10 == 1 and curlyD_max_abs_diff < tol and curlyY_max_abs_diff < tol:
            return curlyDs_sum, curlyYs_sum, i
        if i == maxit - 1:
            raise ValueError(f'No convergence of asymptotic discounted sums after {maxit} iterations!')


def asymp_disc_sum_curlyE(ss, outputs, policy, demean=True, ss_policy_repr=None, verbose=False, maxit=1000, tol=1e-8):
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
        if i % 10 == 1 and verbose:
            print(f"Iteration {i} max abs change in curlyE sum is {curlyE_max_abs_diff}")
        if i % 10 == 1 and curlyE_max_abs_diff < tol:
            return curlyEs_sum, i
        if i == maxit - 1:
            raise ValueError(f'No convergence of asymptotic discounted sums after {maxit} iterations!')


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
