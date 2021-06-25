"""Various utility functions to ease computation"""


# TODO: This function currently serves two separate purposes.
#   1) When `inputs` is a list, then make_inputs is evaluated at all of the steady state values
#      of the inputs, which is generally used in numerically differentiating from steady state
#   2) When `inputs` is a dict, then make_inputs assists in priming the back_iter_vars and exogenous
#      variables contained within `inputs` for standard evaluations of the back_step_fun
#   Functionality 1) is used in the general code, whereas functionality 2) is meant to make the SIM model
#   code compatible with the SSJ-like functions of "priming" the back_iter_vars and exogenous.
#   This is not an ideal solution, so when we get rid of the "priming" convention, we should dispense with this.
# To mirror the functionality of the same-named method in HetBlock
# Hopefully will do away with priming with literal "_p" string appending later on
def make_inputs(inputs, ss, back_iter_vars, exogenous):
    inputs_dict = {}
    for inp in inputs:
        if inp in back_iter_vars or inp in exogenous:
            inputs_dict[inp + "_p"] = inputs[inp] if isinstance(inputs, dict) else ss[inp]
        else:
            inputs_dict[inp] = inputs[inp] if isinstance(inputs, dict) else ss[inp]
    return inputs_dict
