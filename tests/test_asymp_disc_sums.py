"""Testing the functionality in the asymptotic discounted sums module"""

import copy
import numpy as np

import src.ssj_template_code.standard_incomplete_markets as sim
import src.ssj_template_code.fake_news as fn
import src.asymp_disc_sums as ads


# Testing modified backward_step that is independent of SSJ against the template code
def test_backward_step_fakenews():
    T = 3
    h = 1e-4
    shocked_inputs = {'r': 1.}  # Here, ex-post r

    ss = sim.steady_state(**sim.example_calibration())

    inputs = ['Va', 'Pi', 'a_grid', 'y', 'r', 'beta', 'eis']
    outputs = ['A', 'C']
    back_iter_vars = ['Va']
    back_iter_outputs = ['Va', 'a', 'c']
    policy = ['a']
    ss_policy_repr = ads.get_sparse_ss_policy_repr(ss, policy)
    outputs_ss_vals = sim.backward_iteration(**{i: ss[i] for i in inputs})

    curlyYs, curlyDs = fn.get_curlyYs_curlyDs(ss, T, shocked_inputs, h=h)

    # The T - 1 shocked inputs
    curlyVs_Tm1, curlyDs_Tm1, curlyYs_Tm1 = ads.backward_step_fakenews(ss, sim.backward_iteration, inputs, outputs,
                                                                       back_iter_vars, back_iter_outputs,
                                                                       policy, shocked_inputs, h=h,
                                                                       ss_policy_repr=ss_policy_repr,
                                                                       outputs_ss_vals=outputs_ss_vals)

    # The T - 2 shocked inputs
    ss_Tm2 = copy.deepcopy(ss)
    ss_Tm2["Va"] = ss["Va"] + curlyVs_Tm1["Va"] * h
    curlyVs_Tm2, curlyDs_Tm2, curlyYs_Tm2 = ads.backward_step_fakenews(ss_Tm2, sim.backward_iteration, inputs, outputs,
                                                                       back_iter_vars, back_iter_outputs,
                                                                       policy, {}, h=h, outputs_ss_vals=outputs_ss_vals)

    # The T - 3 shocked inputs
    ss_Tm3 = copy.deepcopy(ss)
    ss_Tm3["Va"] = ss["Va"] + curlyVs_Tm2["Va"] * h
    curlyVs_Tm3, curlyDs_Tm3, curlyYs_Tm3 = ads.backward_step_fakenews(ss_Tm3, sim.backward_iteration, inputs, outputs,
                                                                       back_iter_vars, back_iter_outputs,
                                                                       policy, {}, h=h, outputs_ss_vals=outputs_ss_vals)

    # For manual checking
    # curlyDs_comp_Tm1 = curlyDs["r"][0]
    # curlyDdiff_Tm1 = curlyDs_Tm1 - curlyDs_comp_Tm1
    # curlyDmaxdiffs_Tm1 = np.max(np.abs(curlyDdiff_Tm1), axis=1)
    #
    # curlyDs_comp_Tm2 = curlyDs["r"][1]
    # curlyDdiff_Tm2 = curlyDs_Tm2 - curlyDs_comp_Tm2
    # curlyDmaxdiffs_Tm2 = np.max(np.abs(curlyDdiff_Tm2), axis=1)

    assert np.isclose(curlyYs_Tm1["A"], curlyYs["A"]["r"][0])
    assert np.isclose(curlyYs_Tm1["C"], curlyYs["C"]["r"][0])
    assert np.all(np.isclose(curlyDs_Tm1, curlyDs["r"][0]))

    assert np.isclose(curlyYs_Tm2["A"], curlyYs["A"]["r"][1])
    assert np.isclose(curlyYs_Tm2["C"], curlyYs["C"]["r"][1])
    assert np.all(np.isclose(curlyDs_Tm2, curlyDs["r"][1]))

    assert np.isclose(curlyYs_Tm3["A"], curlyYs["A"]["r"][2])
    assert np.isclose(curlyYs_Tm3["C"], curlyYs["C"]["r"][2])
    assert np.all(np.isclose(curlyDs_Tm3, curlyDs["r"][2]))
