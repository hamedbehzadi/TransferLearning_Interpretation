from torch import Tensor
from torch.linalg import norm
# import numpy as np
# from numpy import norm

class OrthogonalProcrustes(object):
    def __init__(self) -> None:
        pass

    def orthogonal_procrustes_distance(self, x1: Tensor, x2: Tensor, normalize: bool = False) -> Tensor:
        """
        Computes the orthoginal procrustes distance.
        If normalized then the answer is divided by 2 so that it's in the interval [0, 1].

        Expected input:
            - two matrices e.g.
                - two weight matrices of size [num_weights1, num_weights2]
                - or two matrices of activations [batch_size, dim_of_layer] (used by paper [1])

        d_proc(A*, B) = ||A||^2_F + ||B||^2_F - 2||A^T B||_*
        || . ||_* = nuclear norm = sum of singular values sum_i sig(A)_i = ||A||_*

        Note: - this only works for matrices. So it's works as a metric for FC and transformers (or at least previous work
        only used it for transformer [1] which have FC and no convolutions.
        - note,

        ref:
        - [1] https://arxiv.org/abs/2108.01661
        - [2] https://discuss.pytorch.org/t/is-there-an-orthogonal-procrustes-for-pytorch/131365
        - [3] https://ee227c.github.io/code/lecture5.html#nuclear-norm

        :param x1:
        :param x2:
        :return:
        """
        # x1x2 = torch.bmm(x1, x2)
        x1x2 = x1.t() @ x2
        d: Tensor = norm(x1, 'fro') + norm(x2, 'fro') - 2 * norm(x1x2, 'nuc')
        d: Tensor = d / 2.0 if normalize else d
        return d

    def orthogonal_procrustes_similarity(self, x1: Tensor, x2: Tensor, normalize: bool = False) -> Tensor:
        """
        Returns orthogonal procurstes similarity. If normalized then output is in invertval [0, 1] and if not then output
        is in interval [0, 1]. See orthogonal_procrustes_distance for details and references.

        :param x1:
        :param x2:
        :param normalize:
        :return:
        """
        d = self.orthogonal_procrustes_distance(x1, x2, normalize)
        sim: Tensor = 1.0 - d if normalize else 2.0 - d
        return sim