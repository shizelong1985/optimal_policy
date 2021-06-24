"""A script for implementing the asymptotic discounted sums needed for SS of the optimum calculations"""

import copy
import numpy as np
import matplotlib.pyplot as plt

import src.ssj_template_code.standard_incomplete_markets as sim
import src.ssj_template_code.fake_news as fn
import src.asymp_disc_sums as ads


# Standard way of getting curlyYs, curlyDs, and curlyEs given T
ss = sim.steady_state(**sim.example_calibration())

h = 1e-4
shocked_inputs = {'r': 1.}  # Here, ex-post r

# New way of getting cumulative sums of curlyYs, curlyDs, and curlyEs, solving for tolerances as opposed to
# specifying a given T ex-ante
inputs = ['Va', 'Pi', 'a_grid', 'y', 'r', 'beta', 'eis']
outputs = ['A', 'C']
back_iter_vars = ['Va']
back_iter_outputs = ['Va', 'a', 'c']
policy = ['a']
ss_policy_repr = ads.get_sparse_ss_policy_repr(ss, policy)
outputs_ss_vals = sim.backward_iteration(**{i: ss[i] for i in inputs})


curlyDs_sum, curlyYs_sum, T_endo = ads.asymp_disc_sums(ss, sim.backward_iteration, inputs, outputs,
                                                       back_iter_vars, back_iter_outputs,
                                                       policy, shocked_inputs, h=h,
                                                       ss_policy_repr=ss_policy_repr, outputs_ss_vals=outputs_ss_vals,
                                                       verbose=True)

curlyYs, curlyDs = fn.get_curlyYs_curlyDs(ss, T_endo, shocked_inputs, h=h)

curlyYs_sum_manual, curlyDs_sum_manual = {}, np.empty_like(curlyDs["r"])
for t in range(T_endo):
    curlyDs_sum_manual += ss["beta"] ** -t * curlyDs["r"][t, ...]
    for o in outputs:
        if t == 0:
            curlyYs_sum_manual[o] = ss["beta"] ** -t * curlyYs[o]["r"][t]
        else:
            curlyYs_sum_manual[o] += ss["beta"] ** -t * curlyYs[o]["r"][t]

assert np.all(np.isclose(curlyDs_sum, curlyDs_sum_manual))
for o in outputs:
    assert np.isclose(curlyYs_sum[o], curlyYs_sum_manual[o])
