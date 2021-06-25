"""Testing the functionality in the asymptotic discounted sums module"""

import copy
import numpy as np

import src.ssj_template_code.standard_incomplete_markets as sim
import src.ssj_template_code.fake_news as fn
import src.asymp_disc_sums as ads


# Testing modified backward_step that is independent of SSJ against the template code
def test_backward_step_fakenews(sim_model):
    # Load the model variables
    ss, back_step_fun, inputs, outputs, back_iter_vars, back_iter_outputs, policy, exogenous,\
    ss_policy_repr, outputs_ss_vals = sim_model

    # Test settings
    T = 3
    h = 1e-4
    shocked_inputs = {'r': 1.}  # Here, ex-post r

    curlyYs, curlyDs = fn.get_curlyYs_curlyDs(ss, T, shocked_inputs, h=h)

    # The T - 1 shocked inputs
    curlyVs_Tm1, curlyDs_Tm1, curlyYs_Tm1 = ads.backward_step_fakenews(ss, back_step_fun, inputs, outputs,
                                                                       back_iter_vars, back_iter_outputs,
                                                                       policy, exogenous, shocked_inputs, h=h,
                                                                       ss_policy_repr=ss_policy_repr,
                                                                       outputs_ss_vals=outputs_ss_vals)

    # The T - 2 shocked inputs
    ss_Tm2 = copy.deepcopy(ss)
    ss_Tm2["Va"] = ss["Va"] + curlyVs_Tm1["Va"]["r"] * h
    shocked_inputs = {'r': 0.}  # Since we only want to direct impact of the shock at period T - 1
    curlyVs_Tm2, curlyDs_Tm2, curlyYs_Tm2 = ads.backward_step_fakenews(ss_Tm2, back_step_fun, inputs, outputs,
                                                                       back_iter_vars, back_iter_outputs,
                                                                       policy, exogenous, shocked_inputs, h=h,
                                                                       outputs_ss_vals=outputs_ss_vals)

    # The T - 3 shocked inputs
    ss_Tm3 = copy.deepcopy(ss)
    ss_Tm3["Va"] = ss["Va"] + curlyVs_Tm2["Va"]["r"] * h
    curlyVs_Tm3, curlyDs_Tm3, curlyYs_Tm3 = ads.backward_step_fakenews(ss_Tm3, back_step_fun, inputs, outputs,
                                                                       back_iter_vars, back_iter_outputs,
                                                                       policy, exogenous, shocked_inputs, h=h,
                                                                       outputs_ss_vals=outputs_ss_vals)

    assert np.isclose(curlyYs_Tm1["A"]["r"], curlyYs["A"]["r"][0])
    assert np.isclose(curlyYs_Tm1["C"]["r"], curlyYs["C"]["r"][0])
    assert np.all(np.isclose(curlyDs_Tm1["r"], curlyDs["r"][0]))

    assert np.isclose(curlyYs_Tm2["A"]["r"], curlyYs["A"]["r"][1])
    assert np.isclose(curlyYs_Tm2["C"]["r"], curlyYs["C"]["r"][1])
    assert np.all(np.isclose(curlyDs_Tm2["r"], curlyDs["r"][1]))

    assert np.isclose(curlyYs_Tm3["A"]["r"], curlyYs["A"]["r"][2])
    assert np.isclose(curlyYs_Tm3["C"]["r"], curlyYs["C"]["r"][2])
    assert np.all(np.isclose(curlyDs_Tm3["r"], curlyDs["r"][2]))


def test_curlyYs_and_curlyDs_sums(sim_model):
    # Load the model variables
    ss, back_step_fun, inputs, outputs, back_iter_vars, back_iter_outputs, policy, exogenous, ss_policy_repr,\
    outputs_ss_vals = sim_model

    # Test settings
    h = 1e-4
    shocked_inputs = {'r': 1.}  # Here, ex-post r

    curlyDs_sum, curlyYs_sum, T_endo = ads.asymp_disc_sums_curlyDs_and_Ys(ss, back_step_fun, inputs, outputs,
                                                                          back_iter_vars, back_iter_outputs,
                                                                          policy, exogenous, shocked_inputs, h=h,
                                                                          ss_policy_repr=ss_policy_repr,
                                                                          outputs_ss_vals=outputs_ss_vals)

    curlyYs, curlyDs = fn.get_curlyYs_curlyDs(ss, T_endo, shocked_inputs, h=h)

    curlyYs_sum_manual, curlyDs_sum_manual = {}, np.zeros_like(curlyDs["r"])
    for t in range(T_endo):
        curlyDs_sum_manual += ss["beta"] ** -t * curlyDs["r"][t, ...]
        for o in outputs:
            if t == 0:
                curlyYs_sum_manual[o] = ss["beta"] ** -t * curlyYs[o]["r"][t]
            else:
                curlyYs_sum_manual[o] += ss["beta"] ** -t * curlyYs[o]["r"][t]

    assert np.all(np.isclose(curlyDs_sum["r"], curlyDs_sum_manual))
    for o in outputs:
        assert np.isclose(curlyYs_sum[o]["r"], curlyYs_sum_manual[o])


def test_curlyEs_sums(sim_model):
    # Load the model variables
    ss, _, _, outputs, _, _, policy, _, ss_policy_repr, _ = sim_model

    # Test ergodicity of curlyEs
    T = 1000
    curlyEs = fn.get_curlyEs(ss, T)
    curlyEs_A_at_T = curlyEs["A"][-1, ...]
    assert np.all(np.isclose(curlyEs_A_at_T, np.vdot(ss["D"], ss["a"])))

    curlyEs_sum, T_endo = ads.asymp_disc_sum_curlyE(ss, outputs, policy, demean=True, ss_policy_repr=ss_policy_repr)

    curlyEs = fn.get_curlyEs(ss, T_endo)
    curlyEs_sum_manual = {}
    for t in range(T_endo):
        for o in outputs:
            if t == 0:
                curlyEs_sum_manual[o] = ss["beta"] ** t * (curlyEs[o][t] - np.vdot(ss["D"], ss[o.lower()]))
            else:
                curlyEs_sum_manual[o] += ss["beta"] ** t * (curlyEs[o][t] - np.vdot(ss["D"], ss[o.lower()]))

    for o in outputs:
        assert np.all(np.isclose(curlyEs_sum[o], curlyEs_sum_manual[o]))


def test_asymp_disc_sums(sim_model):
    # Load the model variables
    ss, back_step_fun, inputs, outputs, back_iter_vars, back_iter_outputs, policy, exogenous, ss_policy_repr,\
    outputs_ss_vals = sim_model

    # Test settings
    h = 1e-4
    shocked_inputs = {'r': 1., 'tau': 1.}  # Here, ex-post r

    asymp_disc_sums = ads.asymp_disc_sums(ss, sim.backward_iteration, inputs, outputs,
                                          back_iter_vars, back_iter_outputs, policy, exogenous,
                                          shocked_inputs, h=h, ss_policy_repr=ss_policy_repr,
                                          outputs_ss_vals=outputs_ss_vals)

    bigT = 1000
    Js = fn.jacobians(ss, bigT, shocked_inputs)

    bigS = 800
    beta_vec = np.empty(bigT)
    beta_vec[:bigS] = ss["beta"] ** np.flip(-np.arange(bigS) - 1)
    beta_vec[bigS:] = ss["beta"] ** np.arange(bigT - bigS)

    asymp_disc_sums_from_Js = {}
    for o in outputs:
        asymp_disc_sums_from_Js[o] = {s: np.vdot(beta_vec, Js[o][s][:, bigS]) for s in shocked_inputs.keys()}

    for o in outputs:
        for s in shocked_inputs.keys():
            # TODO: Try to see if we can more systematically test tolerances choosing bigT, bigS, and tol in asymp_disc_sums
            # print(f"Difference for output {o} and shock {s} is {np.abs(asymp_disc_sums[o][s] - asymp_disc_sums_from_Js[o][s])}")
            assert np.abs(asymp_disc_sums[o][s] - asymp_disc_sums_from_Js[o][s]) < 1e-4

