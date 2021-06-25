"""The simplest heterogeneous agent Ramsey problem"""

import numpy as np

import src.asymp_disc_sums as ads


def simple_het_ramsey_resid(ss, block_inputs, shocked_inputs, ss_policy_repr, outputs_ss_vals, **kwargs):
    back_step_fun, inputs, outputs, back_iter_vars, back_iter_outputs, policy, exogenous = block_inputs

    sums = ads.asymp_disc_sums(ss, back_step_fun, inputs, outputs, back_iter_vars,
                               back_iter_outputs, policy, exogenous, shocked_inputs, ss_policy_repr=ss_policy_repr,
                               outputs_ss_vals=outputs_ss_vals, **kwargs)
    curlyE_b_tau, curlyE_logb_r = -sums["A"]["w"], sums["A"]["r"]/ss["A"]

    uc = ss["Va"] / (1 + ss["r"])
    lambda_hh_e = np.einsum("sb, s, sb", uc, ss["e_grid"], ss["D"])
    lambda_hh_b = np.einsum("sb, b, sb", uc, ss["a_grid"], ss["D"]) / ss["A"]

    spread = 1 - ss["beta"] * (1 + ss["r"])

    target = lambda_hh_e * (1 - spread / ss["beta"] * curlyE_logb_r) - lambda_hh_b * (1 + spread * curlyE_b_tau)
    return target
