from slicetca.invariance.iterative_invariance import sgd_invariance, within_invariance
from slicetca.invariance.analytic_invariance import svd_basis
from slicetca.invariance.criteria import *
from slicetca.core.decompositions import SliceTCA

dict_L2_invariance_objectives = {'orthogonality': orthogonality_component_type_wise,
                                 'L2': l2,
                                 'peaks': peak_coincidence,
                                 'dtw': dtw,
                                 'soft_dtw': soft_dtw,
                                 'skewness': skewness}

dict_L3_invariance_functions = {'svd': svd_basis,
                                'orthogonality': model_ortho,
                                'skewness': skewness}


def invariance(model: SliceTCA,
               L2: str = 'orthogonality',
               L3: str = 'svd',
               **kwargs):
    """
    High level function for invariance optimization.
    Note: modifies inplace, deepcopy your model if you want a copy of the not invariance-optimized components.

    :param model: A sliceTCA model.
    :param L2: String, currently only supports 'regularization', you may add additional objectives.
    :param L3: String, currently only supports 'svd'.
    :param kwargs: Key-word arguments to be passed to L2 and L3 optimization functions. See iterative_function.py
    :return: model with modified components.
    """

    if sum([r!=0 for r in model.ranks])>1 and L2 is not None:
        model = sgd_invariance(model, objective_function=dict_L2_invariance_objectives[L2], **kwargs)
    elif L2 is not None:
        model = within_invariance(model, objective_function=dict_L2_invariance_objectives[L2], **kwargs)
    if L3 is not None:
        model = dict_L3_invariance_functions[L3](model)

    return model
