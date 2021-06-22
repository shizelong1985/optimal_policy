"""A script for implementing the asymptotic discounted sums needed for SS of the optimum calculations"""

import numpy as np
import matplotlib.pyplot as plt

import src.ssj_template_code.standard_incomplete_markets as sim
import src.ssj_template_code.fake_news as fn
import src.asymp_disc_sums as ads


# Standard way of getting curlyYs, curlyDs, and curlyEs given T
ss = sim.steady_state(**sim.example_calibration())

T = 300
shocks = {"r": 1}  # Here, ex-post r

curlyYs, curlyDs = fn.get_curlyYs_curlyDs(ss, T, shocks)
curlyEs = fn.get_curlyEs(ss, T)

# The T - 1 shocked inputs
h = 1e-4
shocked_inputs = {"r": ss["r"] + h}
curlyYs_Tm1, curlyDs_Tm1 = ads.get_curlyYs_curlyDs_single_step(ss, shocked_inputs, h=h)

assert np.isclose(curlyYs_Tm1["A"], curlyYs["A"]["r"][0])
assert np.isclose(curlyYs_Tm1["C"], curlyYs["C"]["r"][0])
assert np.all(np.isclose(curlyDs_Tm1, curlyDs["r"][0]))
