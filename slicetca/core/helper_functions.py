import torch
import itertools


def squared_difference(x, x_hat, weights=None):
    out = (x - x_hat) ** 2
    if weights is not None:
        out *= weights / weights.mean()
    return out


def poisson_log_likelihood(spikes, rates, spikes_factorial, activation=torch.nn.functional.softplus):

    likelihood = torch.exp(-activation(rates)) * torch.pow(activation(rates), spikes) / spikes_factorial

    return -torch.log(likelihood)


def to_sparse(x: torch.Tensor, mask: torch.Tensor):
    """
    Converts a dense tensor to a sparse tensor based on a mask. The mask should
    be a boolean tensor with the same shape as the dense, and the resulting sparse
    tensor contains only subset of the elements of the original tensor where the
    mask is True.

    :param x: torch.Tensor
    :param mask: torch.Tensor
    :return: torch.sparse_coo_tensor
    """
    # Create a mask for the original tensor
    orig_mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
    orig_mask[mask] = True

    # # check if the mask allows for sparse compression in any particular dimension
    # if mask.any(dim=0).all():
    #     out = torch.sparse_compressed_tensor()

    # Create the new coo tensor
    out = torch.sparse_coo_tensor(orig_mask.nonzero().t(),
                                  x[mask],
                                  x.shape,
                                  device=x.device,
                                  requires_grad=True,
                                  is_coalesced=True)
    try: out = out.to_sparse_csr(1)
    except: pass

    return out


def generate_square_wave_tensor(*dims):
    tensor = torch.zeros(dims, dtype=torch.float64)
    wave_length = dims[-1] // 2

    for idx in itertools.product(*[range(d) for d in dims[:-1]]):
        start = (sum(idx) * wave_length) % dims[-1]
        end = start + wave_length
        tensor[idx + (slice(start, end),)] = 1

    return tensor


def subselect(coo_tensor: torch.sparse_coo_tensor, mask: torch.Tensor):
    """
    Subselects a sparse tensor based on a mask. The mask should be a boolean
    tensor with the same shape as the sparse, and the resulting sparse tensor
    contains only subset of the elements of the original tensor where the mask
    is True.

    :param coo_tensor: torch.sparse_coo_tensor
    :param mask: torch.Tensor
    :return: torch.sparse_coo_tensor
    """
    # Create a mask for the original tensor
    coo_tensor = coo_tensor.coalesce() if not coo_tensor.is_coalesced() else coo_tensor
    orig_mask = torch.zeros(coo_tensor.shape, dtype=torch.bool, device=coo_tensor.device)
    orig_mask[*coo_tensor.indices()] = True

    # Combine the mask with the original mask
    combined_mask = mask & orig_mask

    # Create the new coo tensor
    out = torch.sparse_coo_tensor(combined_mask.nonzero().t(),
                                  coo_tensor.to_dense()[combined_mask],
                                  coo_tensor.shape,
                                  device=coo_tensor.device,
                                  requires_grad=True,
                                  is_coalesced=True)

    return out

