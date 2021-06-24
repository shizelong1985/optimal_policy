"""A script for implementing the asymptotic discounted sums needed for SS of the optimum calculations"""

import copy
import numpy as np
import matplotlib.pyplot as plt

import src.ssj_template_code.standard_incomplete_markets as sim
import src.ssj_template_code.fake_news as fn
import src.asymp_disc_sums as ads


# Standard way of getting curlyYs, curlyDs, and curlyEs given T
ss = sim.steady_state(**sim.example_calibration())

T = 300
# h = 1e-4
h = 1e-5
shocks = {'r': 1.}  # Here, ex-post r

curlyYs, curlyDs = fn.get_curlyYs_curlyDs(ss, T, shocks, h=h)
curlyEs = fn.get_curlyEs(ss, T)

# New way of getting cumulative sums of curlyYs, curlyDs, and curlyEs, solving for tolerances as opposed to
# specifying a given T ex-ante
inputs = ['Va', 'Pi', 'a_grid', 'y', 'r', 'beta', 'eis']
outputs = ['A', 'C']
back_iter_vars = ['Va']
back_iter_outputs = ['Va', 'a', 'c']
policy = ['a']
# shocked_inputs = {'r': ss['r'] + h}
shocked_inputs = {'r': 1.}
policy_repr = ads.get_sparse_policy_repr(ss, policy)
outputs_ss_vals = sim.backward_iteration(**{i: ss[i] for i in inputs})

# The T - 1 shocked inputs
curlyVs_Tm1, curlyDs_Tm1, curlyYs_Tm1 = ads.backward_step_fakenews(ss, inputs, outputs, back_iter_vars, back_iter_outputs,
                                                                   policy, shocked_inputs, h=h,
                                                                   policy_repr=policy_repr, outputs_ss_vals=outputs_ss_vals)

# The T - 2 shocked inputs
ss_Tm2 = copy.deepcopy(ss)
ss_Tm2["Va"] = ss["Va"] + curlyVs_Tm1["Va"] * h
shocked_inputs_Tm2 = {}  # Probably a more efficient way to implement in the code to not calculate a shock
curlyVs_Tm2, curlyDs_Tm2, curlyYs_Tm2 = ads.backward_step_fakenews(ss_Tm2, inputs, outputs, back_iter_vars, back_iter_outputs,
                                                                   policy, {}, h=h,
                                                                   policy_repr=policy_repr, outputs_ss_vals=outputs_ss_vals)

curlyDs_comp_Tm1 = curlyDs["r"][0]
curlyDdiff_Tm1 = curlyDs_Tm1 - curlyDs_comp_Tm1
curlyDmaxdiffs_Tm1 = np.max(np.abs(curlyDdiff_Tm1), axis=1)

curlyDs_comp_Tm2 = curlyDs["r"][1]
curlyDdiff_Tm2 = curlyDs_Tm2 - curlyDs_comp_Tm2
curlyDmaxdiffs_Tm2 = np.max(np.abs(curlyDdiff_Tm2), axis=1)

assert np.isclose(curlyYs_Tm1["A"], curlyYs["A"]["r"][0])
assert np.isclose(curlyYs_Tm1["C"], curlyYs["C"]["r"][0])
assert np.all(np.isclose(curlyDs_Tm1, curlyDs["r"][0]))

assert np.isclose(curlyYs_Tm2["A"], curlyYs["A"]["r"][1])
assert np.isclose(curlyYs_Tm2["C"], curlyYs["C"]["r"][1])
assert np.all(np.isclose(curlyDs_Tm2, curlyDs["r"][1]))

aaa
for i in range(len(ss["y"])):
    plt.plot(curlyDdiff[i, :])
plt.show()

for i in range(len(ss["y"])):
    plt.plot(curlyDs_Tm1[i, :])
plt.show()

for i in range(len(ss["y"])):
    plt.plot(curlyDs["r"][0][i, :])
plt.show()
