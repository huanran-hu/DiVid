import numpy as np

from divid.math_utils import pairwise_cosine_distance, rbf_metrics, upper_triangle_mean


def test_upper_triangle_mean_ignores_diagonal():
    matrix = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 5.0], [3.0, 5.0, 0.0]])
    assert upper_triangle_mean(matrix) == 3.0


def test_cosine_distance_identical_vectors_is_zero():
    distance, mean = pairwise_cosine_distance(np.ones((3, 4)))
    assert np.allclose(distance, 0.0)
    assert mean == 0.0


def test_rbf_kernel_is_symmetric_and_unit_diagonal():
    result = rbf_metrics(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]), (1.0, 2.0), 1.0)
    assert np.allclose(result["similarity"], result["similarity"].T)
    assert np.allclose(np.diag(result["similarity"]), 1.0)
