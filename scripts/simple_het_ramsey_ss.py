"""Replicating/extending the preliminary results in the `simplest possible het agent ramsey steady state` notebook"""

import copy
import numpy as np
import matplotlib.pyplot as plt

import sequence_jacobian as sj
import sequence_jacobian.models.krusell_smith as ks
from src.models.simple_het import simple_het_ramsey_resid
from src.ramsey_steady_state import solve_ramsey_steady_state

# Setting up input arguments for Ramsey steady state solution
e_grid, pi, Pi = sj.markov_rouwenhorst(0.9, 0.9)
a_grid = sj.agrid(10_000, 300)

tau = 0.
calibration = {"Pi": Pi, "e_grid": e_grid, "a_grid": a_grid, "eis": 0.2, "beta": 0.95, "w": 1 - tau}
calibration["Va"] = 1/(e_grid[:, np.newaxis] + 0.1 * a_grid[np.newaxis, :])
unknowns = {"r": (-0.05, 1/calibration["beta"] - 1 - 0.002)}
ss_fun = ks.household.steady_state
resid_fun = simple_het_ramsey_resid

# This is standing in for what will eventually just be the blocks in the DAG that follow the solution of
# a given unknown. In this case, we solve for tau after the fact, since it just scales the problem.
def post_process_fun(ss):
    tau = ss["r"] * ss["A"] / (1 + ss["r"] * ss["A"])
    return tau

hh = ks.household
# Note that *must* provide back_step_output_list in the right order or the numerical differentiation gets screwed up.
block_inputs = (hh.back_step_fun, hh.inputs | {"Va"}, hh.outputs,
                hh.back_iter_vars, hh.back_step_output_list, hh.policy, hh.exogenous)
# Since we don't have tau in the household back_step_fun, we use a negative unit shock to w instead, which is equivalent
shocked_inputs = {"r": 1., "w": 1.}
resid_kwargs = {"tol": 1e-5}  # Could choose the tol and maxit for the asymptotic discounted sums
optim_kwargs = {"xtol": 1e-8}  # Could choose xtol, rtol, and maxiter

# ss = solve_ramsey_steady_state(calibration, unknowns, ss_fun, resid_fun, post_process_fun, block_inputs,
#                                shocked_inputs, resid_kwargs=resid_kwargs, optim_kwargs=optim_kwargs)

# Testing
import src.asymp_disc_sums as ads
from src.ramsey_steady_state import steady_state
back_step_fun, inputs, outputs, back_iter_vars, back_iter_outputs, policy, exogenous = block_inputs
ss = copy.deepcopy(calibration)
# ss.update({"r": -0.05})
ss.update({"r": 1/calibration["beta"] - 1 - 0.002})
# ss.update({"r": -0.02321815157081399})
ss.update(steady_state(ss, ss_fun))

# subset = 100
# for i in range(len(ss["e_grid"])):
#     # plt.plot(ss["a_grid"][:subset], ss["a"][0, :subset])
#     # plt.plot(ss["a_grid"][:subset], ss["c"][0, :subset])
#     plt.plot(ss["a_grid"][:subset], ss["Va"][0, :subset])
# plt.show()
ss_policy_repr = ads.get_sparse_ss_policy_repr(ss, policy)
outputs_ss_vals = tuple(ss[i] for i in back_iter_outputs)
curlyDs, curlyYs, T_endo = ads.asymp_disc_sums_curlyDs_and_Ys(ss, back_step_fun, inputs, outputs, back_iter_vars,
                                                              back_iter_outputs, policy, exogenous, shocked_inputs,
                                                              h=1e-4, recompute_policy_grid=True, ss_policy_repr=ss_policy_repr,
                                                              outputs_ss_vals=outputs_ss_vals, verbose=True,
                                                              # checkit=1, maxit=100, tol=1e-5)
                                                              checkit=10, maxit=1000, tol=1e-5)

# # w/ r = -0.05 checks out against stats(ss, T=200) from the notebook, roughly accurate to 1e-5
# sums = ads.asymp_disc_sums(ss, back_step_fun, inputs, outputs, back_iter_vars,
#                            back_iter_outputs, policy, exogenous, shocked_inputs, ss_policy_repr=ss_policy_repr,
#                            outputs_ss_vals=outputs_ss_vals, verbose=True, maxit=1000, tol=1e-6)
# curlyE_b_tau, curlyE_logb_r = -sums["A"]["w"], sums["A"]["r"] / ss["A"]


# # Trying the alternative way to calculate asymptotic discounted sums
# from sequence_jacobian.steady_state.classes import SteadyStateDict
#
# bigT = 1000
# bigS = 200
#
# # Only works for a single block as of now
# def construct_ssj_ss_from_flat_ss(ss, block_name, block_internals):
#     toplevel = {i: ss[i] for i in ss if i not in block_internals}
#     internal = {block_name: {i: ss[i] for i in block_internals}}
#     return SteadyStateDict(toplevel, internal=internal)
#
# ss_ssj = construct_ssj_ss_from_flat_ss(ss, "household", ["Va", "D", "a", "c"])
#
# # Janky implementation...
# def jac_fun(ss, bigT, shocked_inputs):
#     return ks.household.jacobian(ss, list(shocked_inputs.keys()), T=bigT).nesteddict
#
# Js = jac_fun(ss_ssj, bigT, shocked_inputs)
#
# # w/ r = -0.05 matches against stats(ss, T=200) from the notebook, roughly accurate to 1e-5
# # w/ r = 1/beta - 1 - 0.002 matches against stats(ss, T=200) from the notebook, accurate to 1e0
# sums_from_Js = ads.asymp_disc_sums_from_Js(ss_ssj, jac_fun, outputs, shocked_inputs, bigT=bigT, bigS=bigS)
# curlyE_b_tau_from_Js, curlyE_logb_r_from_Js = -sums_from_Js["A"]["w"], sums_from_Js["A"]["r"] / ss["A"]
