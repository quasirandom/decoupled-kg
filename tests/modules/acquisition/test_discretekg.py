import math
import re

import pytest
import torch

from decoupledbo.modules.acquisition.discretekg import (
    DiscreteKnowledgeGradient,
    calculate_discrete_kg,
    calculate_discrete_kg_conditioning_on_single_output,
    calculate_epigraph_indices,
    calculate_expected_value_of_piecewise_linear_function,
)
from tests.utils import torch_assert_close


@pytest.fixture()
def discretisation():
    n = 3  # mesh granularity
    return torch.stack(
        [
            torch.repeat_interleave(torch.linspace(0, 1, n), n),
            torch.tile(torch.linspace(0, 1, n), (n,)),
        ]
    ).T


class TestDiscreteKnowledgeGradient:
    @pytest.fixture()
    def target_x(self):
        # Define a test x with t-batch shape 2x3 and q=1
        return torch.tensor(
            [
                [
                    [[0.5, 0.5]],
                    [[0, 1]],
                    [[0, 0.5]],
                ],
                [
                    [[0, 0]],
                    [[1, 0]],
                    [[0.5, 0]],
                ],
            ],
        )

    # TODO: Test with automatic discretisation
    # TODO: Test gradients with finite difference

    def test_smoke_test_with_explicit_discretisation(
        self, noisy_model, scalarisation_weights_trio, discretisation, target_x
    ):
        acqf = DiscreteKnowledgeGradient(
            noisy_model,
            x_discretisation=discretisation,
            scalarisation_weights=scalarisation_weights_trio,
        )

        with torch.no_grad():
            kg = acqf(target_x)

        expected_kg = torch.tensor([[0.0383, 0.0224, 0.0130], [0.0005, 0.0058, 0.0015]])
        torch_assert_close(kg, expected_kg, atol=1e-4, rtol=1e-3)

    def test_smoke_test_conditioning_on_single_output_with_explicit_discretisation(
        self, noisy_model, scalarisation_weights_trio, discretisation, target_x
    ):
        acqf = DiscreteKnowledgeGradient(
            noisy_model,
            x_discretisation=discretisation,
            scalarisation_weights=scalarisation_weights_trio,
            target_output_ix=0,
        )

        with torch.no_grad():
            kg = acqf(target_x)

        expected_kg = torch.tensor([[0.0297, 0.0084, 0.0048], [0.0002, 0.0030, 0.0006]])
        torch_assert_close(kg, expected_kg, atol=1e-4, rtol=1e-3)


class TestCalculateDiscreteKg:
    # Tests for
    #   - calculate_discrete_kg
    #   - calculate_discrete_kg_conditioning_on_single_output

    def test_smoke_test(self, noisy_model, scalarisation_weights_trio, discretisation):
        xnew = torch.tensor([0.5, 0.5])
        with torch.no_grad():
            kg = calculate_discrete_kg(
                noisy_model, xnew, discretisation, scalarisation_weights_trio
            )
        assert kg.item() == pytest.approx(0.038261974207699244)

    def test_smoke_test_conditioning_on_single_output(
        self, noisy_model, scalarisation_weights_trio, discretisation
    ):
        xnew = torch.tensor([0.5, 0.5])
        with torch.no_grad():
            obj_idx_new = 0
            kg = calculate_discrete_kg_conditioning_on_single_output(
                noisy_model,
                xnew,
                obj_idx_new,
                discretisation,
                scalarisation_weights_trio,
            )
        assert kg.item() == pytest.approx(0.02968190595713936)

    def test_gradients(self, model, scalarisation_weights, discretisation):
        # The gradients do not work at x values which lie on the boundary of changing
        # which lines make up the epi-graph. For that reason, we avoid (0.5, 0.5).
        xnew = torch.tensor([0.51, 0.51], requires_grad=True)

        # I think autograd only checks the gradients of inputs with requires_grad=True
        torch.autograd.gradcheck(
            calculate_discrete_kg,
            (model, xnew, discretisation, scalarisation_weights),
            raise_exception=True,
        )

    @pytest.mark.parametrize("obj_idx_new", [0, 1])
    def test_gradients_conditioning_on_single_output(
        self, model, scalarisation_weights, discretisation, obj_idx_new
    ):
        # The gradients do not work at x values which lie on the boundary of changing
        # which lines make up the epi-graph. For that reason, we avoid (0.5, 0.5).
        xnew = torch.tensor([0.51, 0.51], requires_grad=True)

        # I think autograd only checks the gradients of inputs with requires_grad=True
        torch.autograd.gradcheck(
            calculate_discrete_kg_conditioning_on_single_output,
            (model, xnew, obj_idx_new, discretisation, scalarisation_weights),
            raise_exception=True,
        )


class TestCalculateEpigraphIndices:
    def test_raises_on_empty_input(self):
        empty_intercepts = torch.tensor([])
        empty_slopes = torch.tensor([])

        expected_msg = (
            "Expected inputs to specify at least one line. "
            "Got intercepts.shape[-1]=0."
        )
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            calculate_epigraph_indices(empty_intercepts, empty_slopes)

    def test_with_zero_slopes(self):
        # Tests the shortcut when all slopes are zero
        intercepts = torch.tensor([1, 1.5])
        slopes = torch.tensor([0, 0])

        indices, intersections = calculate_epigraph_indices(intercepts, slopes)

        torch_assert_close(indices, torch.tensor([1]))
        torch_assert_close(intersections, torch.tensor([]))

    def test_with_single_line(self):
        intercepts = torch.tensor([1.5])
        slopes = torch.tensor([-1.9])

        indices, intersections = calculate_epigraph_indices(intercepts, slopes)

        torch_assert_close(indices, torch.tensor([0]))
        torch_assert_close(intersections, torch.tensor([]))

    @pytest.mark.parametrize("ordered", [True, False])
    def test_with_two_lines(self, ordered, dtype):
        intercepts = torch.tensor([1.5, 0])
        slopes = torch.tensor([-0.5, 0])

        if not ordered:
            intercepts = torch.flip(intercepts, dims=[0])
            slopes = torch.flip(slopes, dims=[0])

        indices, intersections = calculate_epigraph_indices(intercepts, slopes)

        expected_indices = torch.tensor([0, 1] if ordered else [1, 0])
        torch_assert_close(indices, expected_indices)
        torch_assert_close(intersections, torch.tensor([3], dtype=dtype))

    def test_with_two_equal_slopes(self):
        # This test is specifically to catch a bug where two slopes are equal and we
        # were incorrectly assuming the intersections were sorted. Therefore, we set up
        # two lines with equal slopes, followed by two lines with increasing slopes but
        # decreasing intersection order. The second line should be chosen but, before
        # fixing the bug, the first one was being chosen.
        intercepts = torch.tensor([0, 0, -0.5, 0])
        slopes = torch.tensor([-1, -1, 0, 1.5])

        indices, intersections = calculate_epigraph_indices(intercepts, slopes)

        torch_assert_close(indices, torch.tensor([0, 3]))
        torch_assert_close(intersections, torch.tensor([0.0]))

    @pytest.mark.parametrize(
        ("input_order", "expected_indices"),
        [
            pytest.param([0, 1, 2], [0, 2], id="ordered"),
            pytest.param([1, 2, 0], [2, 1], id="unordered"),
        ],
    )
    def test_ignores_lines_below_epigraph(self, input_order, expected_indices, dtype):
        intercepts = torch.tensor([0, -1, 0])
        slopes = torch.tensor([-2, -1, 0])

        intercepts = intercepts[input_order]
        slopes = slopes[input_order]

        indices, intersections = calculate_epigraph_indices(intercepts, slopes)

        torch_assert_close(indices, torch.tensor(expected_indices))
        torch_assert_close(intersections, torch.tensor([0], dtype=dtype))

    @pytest.mark.parametrize(
        "slopes",
        [
            pytest.param([-0.5, 0], id="normal"),
            pytest.param([0, 1e-12], id="tiny-slopes"),
            pytest.param([-0.5, -0.5], id="identical-slopes"),
        ],
    )
    def test_gradients(self, slopes):
        intercepts = torch.tensor([1.5, 0], requires_grad=True)
        slopes = torch.tensor(slopes, requires_grad=True)

        # The function calculate_epigraph_indices returns indices and intersections. We
        # only want to check gradients on the intersections.
        torch.autograd.gradcheck(
            lambda *args: calculate_epigraph_indices(*args)[1],
            (intercepts, slopes),
            raise_exception=True,
        )

    @pytest.mark.parametrize("offset", [0, 1])
    def test_gradients_with_two_of_four_slopes_identical(self, offset):
        # This is to test the gradient calculation with the same set-up as in
        # test_with_two_equal_slopes. The test cannot be done using autograd.gradcheck
        # because the shape of the output tensor changes as we vary epsilon.
        intercepts = torch.tensor([offset, offset, -0.5, 0], requires_grad=True)
        slopes = torch.tensor([-1, -1, 0, 1.5], requires_grad=True)

        indices, intersections = calculate_epigraph_indices(intercepts, slopes)

        only_intersection = intersections.squeeze(0)
        assert only_intersection.ndim == 0

        [grad_slopes] = torch.autograd.grad(
            only_intersection, slopes, retain_graph=True
        )
        [grad_intercepts] = torch.autograd.grad(
            only_intersection, intercepts, retain_graph=True
        )

        torch_assert_close(
            grad_slopes, torch.tensor([0.16 * offset, 0.0, 0.0, -0.16 * offset])
        )
        torch_assert_close(grad_intercepts, torch.tensor([0.4, 0.0, 0.0, -0.4]))


class TestCalculateExpectedValueOfPiecewiseLinearFunction:
    def test_raises_on_empty_input(self, dtype):
        empty_intercepts = torch.tensor([], dtype=dtype)
        empty_slopes = torch.tensor([], dtype=dtype)
        empty_boundaries = torch.tensor([], dtype=dtype)

        expected_msg = (
            "Expected inputs to specify at least one line. "
            "Got intercepts.shape[-1]=0."
        )
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            calculate_expected_value_of_piecewise_linear_function(
                empty_intercepts, empty_slopes, empty_boundaries
            )

    def test_with_single_constant_line(self, dtype):
        intercepts = torch.tensor([1.5], dtype=dtype)
        slopes = torch.tensor([0], dtype=dtype)
        boundaries = torch.tensor([], dtype=dtype)

        expected_value = calculate_expected_value_of_piecewise_linear_function(
            intercepts, slopes, boundaries
        )

        assert expected_value == pytest.approx(1.5)

    def test_with_single_sloped_line(self, dtype):
        intercepts = torch.tensor([0], dtype=dtype)
        slopes = torch.tensor([1], dtype=dtype)
        boundaries = torch.tensor([], dtype=dtype)

        expected_value = calculate_expected_value_of_piecewise_linear_function(
            intercepts, slopes, boundaries
        )

        assert expected_value == pytest.approx(0)

    def test_with_relu_line(self, dtype):
        intercepts = torch.tensor([0, 0], dtype=dtype)
        slopes = torch.tensor([0, 1], dtype=dtype)
        boundaries = torch.tensor([0], dtype=dtype)

        expected_value = calculate_expected_value_of_piecewise_linear_function(
            intercepts, slopes, boundaries
        )

        assert expected_value == pytest.approx(1 / math.sqrt(2 * math.pi))

    def test_with_hump(self, dtype):
        # Shape:
        #      /\
        # ----/  \----

        intercepts = torch.tensor([0, 1, 1, 0], dtype=dtype)
        slopes = torch.tensor([0, 1, -1, 0], dtype=dtype)
        boundaries = torch.tensor([-1, 0, 1], dtype=dtype)

        expected_value = calculate_expected_value_of_piecewise_linear_function(
            intercepts, slopes, boundaries
        )

        expected_expected_value = math.erf(1 / math.sqrt(2)) - (
            1 - math.exp(-1 / 2)
        ) * math.sqrt(2 / math.pi)
        assert expected_value == pytest.approx(expected_expected_value)

    def test_gradients(self, dtype):
        # Shape:
        #      /\
        # ----/  \----

        intercepts = torch.tensor([0, 1, 1, 0], dtype=dtype, requires_grad=True)
        slopes = torch.tensor([0, 1, -1, 0], dtype=dtype, requires_grad=True)
        boundaries = torch.tensor([-1, 0, 1], dtype=dtype, requires_grad=True)

        torch.autograd.gradcheck(
            calculate_expected_value_of_piecewise_linear_function,
            (intercepts, slopes, boundaries),
            raise_exception=True,
        )
