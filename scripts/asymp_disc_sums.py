"""A script for implementing the asymptotic discounted sums needed for SS of the optimum calculations"""

import src.ssj_template_code.standard_incomplete_markets as sim
import src.asymp_disc_sums as ads
from src.utils import make_inputs


# Standard way of getting curlyYs, curlyDs, and curlyEs given T
ss = sim.steady_state(**sim.example_calibration())

h = 1e-4
shocked_inputs = {'r': 1., 'tau': 1.}  # Here, ex-post r

# New way of getting cumulative sums of curlyYs, curlyDs, and curlyEs, solving for tolerances as opposed to
# specifying a given T ex-ante
inputs = ['Va', 'Pi', 'a_grid', 'y', 'r', 'beta', 'eis', 'tau']
outputs = ['A', 'C']
back_iter_vars = ['Va']
back_iter_outputs = ['Va', 'a', 'c']
policy = ['a']
exogenous = ['Pi']
ss_policy_repr = ads.get_sparse_ss_policy_repr(ss, policy)
inputs_dict = make_inputs(inputs, ss, back_iter_vars, exogenous)
outputs_ss_vals = sim.backward_iteration(**inputs_dict)


asymp_disc_sums = ads.asymp_disc_sums(ss, sim.backward_iteration, inputs, outputs,
                                      back_iter_vars, back_iter_outputs, policy, exogenous,
                                      shocked_inputs, h=h, ss_policy_repr=ss_policy_repr,
                                      outputs_ss_vals=outputs_ss_vals, verbose=False)
