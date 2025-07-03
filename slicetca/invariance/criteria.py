from itertools import combinations
import torch
from typing import Sequence

# Example of criteria to use for L2 optimization.


def orthogonality_component_type_wise(reconstructed_tensors_of_each_partition: Sequence[torch.Tensor]):
    """
    Penalizes non-orthogonality between the reconstructed tensors of each partition/slicing.

    :param reconstructed_tensors_of_each_partition: The sum of the terms of a given partition/slicing.
    :return: Torch float.
    """

    l = 0
    for combo in combinations(reconstructed_tensors_of_each_partition, 2):
        l += torch.square(torch.sum(combo[0] * combo[1]) / torch.sqrt(torch.sum(combo[0] ** 2)) / torch.sqrt(
            torch.sum(combo[1] ** 2)))
    return l + l2(reconstructed_tensors_of_each_partition)


def orthogonality_along_dim(reconstructed_tensors_of_each_partition: Sequence[torch.Tensor], dim: int = -1):
    """
    Penalizes non-orthogonality between the reconstructed tensors of each partition/slicing.

    :param reconstructed_tensors_of_each_partition: The sum of the terms of a given partition/slicing.
    :return: Torch float.
    """

    l = 0
    for combo in combinations(reconstructed_tensors_of_each_partition, 2):
        l += torch.nn.CosineSimilarity(dim=dim)(combo[0], combo[1]).mean()
    return l



def l2(reconstructed_tensors_of_each_partition: Sequence[torch.Tensor]):
    """
    Classic L_2 regularization, per reconstructed tensors of each partition/slicing.

    :param reconstructed_tensors_of_each_partition: The sum of the terms of a given partition/slicing.
    :return: Torch float.
    """

    l = 0
    for t in reconstructed_tensors_of_each_partition:
        l += (t ** 2).mean()
    return l


def peak_coincidence(reconstructed_tensors_of_each_partition: Sequence[torch.Tensor], axis: int = -1):
    """
    Penalizes coincidence of peaks between the reconstructed tensors of each partition/slicing.

    :param reconstructed_tensors_of_each_partition: The sum of the terms of a given partition/slicing.
    :return: Torch float.
    """

    l = 0
    for combo in combinations(reconstructed_tensors_of_each_partition, 2):
        l += combo[0].shape[axis] - torch.abs(
                soft_arg_max(torch.sum(combo[0], axis)) -
                soft_arg_max(torch.sum(combo[1], axis)))
    return l

try:
    from tslearn.metrics import SoftDTWLossPyTorch, dtw as ts_dtw

    def dtw(reconstructed_tensors_of_each_partition: Sequence[torch.Tensor]):
        """
        Penalizes dynamic time warping between the reconstructed tensors of each partition/slicing.

        :param reconstructed_tensors_of_each_partition: The sum of the terms of a given partition/slicing.
        :return: Torch float.
        """
        if reconstructed_tensors_of_each_partition[0].ndim == 2:
            reconstructed_tensors_of_each_partition = [r[None] for r in
                                                       reconstructed_tensors_of_each_partition]
        parts = [p.permute(0,2,1) for p in reconstructed_tensors_of_each_partition]
        l = 0
        for combo in combinations(parts, 2):
            for i in range(combo[0].shape[0]):
                l += ts_dtw(combo[0][i], combo[1][i], be='pytorch')
        return l / parts[0].shape[0]

    def soft_dtw(reconstructed_tensors_of_each_partition: Sequence[torch.Tensor]):
        """
        Penalizes dynamic time warping between the reconstructed tensors of each partition/slicing.

        :param reconstructed_tensors_of_each_partition: The sum of the terms of a given partition/slicing.
        :return: Torch float.
        """
        if reconstructed_tensors_of_each_partition[0].ndim == 2:
            reconstructed_tensors_of_each_partition = [r[None] for r in
                                                       reconstructed_tensors_of_each_partition]
        parts = [p.permute(0,2,1) for p in reconstructed_tensors_of_each_partition]
        l = 0
        for combo in combinations(parts, 2):
            l += SoftDTWLossPyTorch(normalize=True, gamma=1)(combo[0], combo[1])
        return l

except ImportError:
    def dtw(reconstructed_tensors_of_each_partition: Sequence[torch.Tensor]):
        raise ImportError("Please install tslearn to use dtw criterion.")

    def soft_dtw(reconstructed_tensors_of_each_partition: Sequence[torch.Tensor]):
        raise ImportError("Please install tslearn to use soft_dtw criterion.")


def soft_arg_max(A, mask=None, beta=10, dim=0, epsilon=1e-12):
    '''
        applay softargmax on A and consider mask, return \sum_i ( i * (exp(A_i * beta) / \sum_i(exp(A_i * beta))))
        according to https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
        :param A:
        :param mask:
        :param dim:
        :param epsilon:
        :return:
        '''
    # According to https://discuss.pytorch.org/t/apply-mask-softmax/14212/7
    A_max = torch.max(A, dim=dim, keepdim=True)[0]
    A_exp = torch.exp((A - A_max)*beta)
    if mask is not None:
        A_exp *= mask  # this step masks
    A_softmax = A_exp / (torch.sum(A_exp, dim=dim, keepdim=True) + epsilon)
    indices = torch.arange(start=0, end=A.size()[dim]).float()
    # print(indices.size(), A_softmax.size())
    return torch.matmul(A_softmax, indices)

def skewness(reconstructed_tensors_of_each_partition: Sequence[torch.Tensor]):
    """
    Penalizes skewness of the reconstructed tensors of each partition/slicing.

    :param reconstructed_tensors_of_each_partition: The sum of the terms of a given partition/slicing.
    :return: Torch float.
    """
    l = 0
    for t in reconstructed_tensors_of_each_partition:
        mean = t.mean()
        std = t.std()
        skew = ((t - mean) ** 3).mean() / (std ** 3 + 1e-12)
        l += skew
    return l

def model_ortho(model):
    return model.orthogonality()
