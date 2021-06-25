"""Code for solving the steady state of the Ramsey problem"""

import copy
from scipy.optimize import brentq

from sequence_jacobian.utilities.misc import smart_zip
import src.asymp_disc_sums as ads
import src.ssj_template_code.standard_incomplete_markets as sim


def steady_state(calibration, ss_fun):
    ss = ss_fun(calibration)

    # Unpack things to make them a flat dict to make things compatible with the asymp_disc_sums code,
    # since as of now, trying to maintain some separation from SSJ
    ss_flat = ss.toplevel.copy()
    for i in ss.internal.values():
        ss_flat.update(i)
    return ss_flat


def solve_ramsey_steady_state(calibration, unknowns, ss_fun, resid_fun, post_process_fun,
                              block_inputs, shocked_inputs, resid_kwargs=None, optim_kwargs=None):
    if resid_kwargs is None:
        resid_kwargs = {}

    ss = copy.deepcopy(calibration)
    _, _, _, _, back_iter_outputs, policy, _ = block_inputs

    def residual(unknown_values):
        ss.update(smart_zip(unknowns.keys(), unknown_values))
        ss.update(steady_state(ss, ss_fun))

        ss_policy_repr = ads.get_sparse_ss_policy_repr(ss, policy)
        outputs_ss_vals = tuple(ss[i] for i in back_iter_outputs)

        return resid_fun(ss, block_inputs, shocked_inputs, ss_policy_repr, outputs_ss_vals, **resid_kwargs)

    # TODO: Later generalize beyond 1-d case
    lb, ub = next(iter(unknowns.values()))
    opt_result = brentq(residual, lb, ub, **optim_kwargs)

    ss.update(zip(unknowns.keys(), opt_result))

    return post_process_fun(ss)
