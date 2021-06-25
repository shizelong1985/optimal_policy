"""Fixtures used by tests"""

import pytest

import src.ssj_template_code.standard_incomplete_markets as sim
import src.asymp_disc_sums as ads
from src.utils import make_inputs


@pytest.fixture(scope='session')
def sim_model():
    ss = sim.steady_state(**sim.example_calibration())

    inputs = ['Va', 'Pi', 'a_grid', 'y', 'r', 'beta', 'eis', 'tau']
    outputs = ['A', 'C']
    back_step_fun = sim.backward_iteration
    back_iter_vars = ['Va']
    back_iter_outputs = ['Va', 'a', 'c']
    policy = ['a']
    exogenous = ['Pi']
    ss_policy_repr = ads.get_sparse_ss_policy_repr(ss, policy)

    inputs_dict = make_inputs(inputs, ss, back_iter_vars, exogenous)
    outputs_ss_vals = sim.backward_iteration(**inputs_dict)

    return ss, back_step_fun, inputs, outputs, back_iter_vars, back_iter_outputs, policy, exogenous, ss_policy_repr, outputs_ss_vals

