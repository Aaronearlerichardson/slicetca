from slicetca.core.decompositions import SliceTCA

import torch
from typing import Sequence


def mm(a: torch.Tensor, b: torch.Tensor):
    """
    Performs generalized matrix multiplication (ijq) x (qkl) -> (ijkl)
    :param a: torch
    :param b:
    :return:
    """
    temp1 = [chr(105+i) for i in range(len(a.size()))]
    temp2 = [chr(105+len(a.size())-1+i) for i in range(len(b.size()))]
    indexes1 = ''.join(temp1)
    indexes2 = ''.join(temp2)
    rhs = ''.join(temp1[:-1])+''.join(temp2[1:])
    formula = indexes1+','+indexes2+'->'+rhs
    return torch.einsum(formula,[a,b])


def batch_outer(a: torch.Tensor, b: torch.Tensor):

    formula = outer_formula(a, b)
    return torch.einsum(formula, [a, b])


def outer_formula(*tensors: torch.Tensor):
    """
    Generates an einsum formula for the outer product of any number of input tensors.

    :param tensors: A sequence of input tensors.
    :return: A string representing the einsum formula for the outer product.
    """
    indices_list = []
    current_index = 105  # ASCII for 'i'

    for tensor in tensors:
        indices = [chr(current_index + i) for i in range(len(tensor.size()))]
        indices_list.append(''.join(indices))
        current_index += len(tensor.size())

    input_indices = ','.join(indices_list)
    output_indices = ''.join(indices_list)
    formula = f"{input_indices}->{output_indices}"

    return formula

def construct_per_type(model: SliceTCA, components: Sequence[Sequence[torch.Tensor]]):
    """
    :param model: SliceTCA model.
    :param components: The components to construct.
    :return: Reconstructed tensor.
    """

    temp = [torch.zeros(model.dimensions).to(model.device) for i in range(len(components))]

    for i in range(len(components)):
        for j in range(model.ranks[i]):
            temp[i] += construct_single_component(model, components, i, j)
    return temp


def construct_per_component(model: SliceTCA, components: Sequence[Sequence[torch.Tensor]], ignore: tuple = ()):
    """
    :param model: SliceTCA model.
    :param components: The components to construct.
    :return: Reconstructed tensor.
    """
    out_size = [dim for i, dim in enumerate(model.dimensions) if i not in ignore]
    temp = [torch.zeros(out_size).to(model.device)
            for i in range(len(components))
            for j in range(model.ranks[i])]

    counter = 0
    for i in range(len(components)):
        for j in range(model.ranks[i]):
            temp[counter] += construct_single_component(model, components, i, j, ignore)
            counter += 1

    return temp


def construct_single_component(model: SliceTCA, components: Sequence[Sequence[torch.Tensor]], partition: int, k: int,
                               ignore: tuple = ()):

    temp2 = [model.positive_function(components[partition][q][k])
             for q in range(len(components[partition])) if q not in ignore]
    if len(ignore) == 0:
        formula = model.einsums[partition]
        permutation = model.inverse_permutations[partition]
    else:
        formula = outer_formula(*temp2)
        permutation = edit_permute(model.inverse_permutations[partition], ignore)
    outer = torch.einsum(formula, temp2)
    outer = outer.permute(permutation)

    return outer


def edit_permute(lst, indices):
    result = []
    offset = 0
    indices = set(indices)

    for elem in lst:
        if isinstance(elem, int):
            if elem not in indices:
                result.append(elem - offset)
            else:
                offset += 1
        else:
            new_elem = [el - offset for el in elem if el not in indices]
            offset += len(elem) - len(new_elem)
            result.append(new_elem)

    return result
