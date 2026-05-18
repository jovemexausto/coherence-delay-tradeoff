from __future__ import annotations

import math
import unittest

from useful_memory_horizon.holder_lower_bound_research import Hölder_asymptotic_constant
from useful_memory_horizon.sharp_family import (
    SharpFamilyAuditConfig,
    asymptotic_staleness_constant,
    dirac_uniform_window_staleness,
    gaussian_location_expected_absolute_error,
    gaussian_location_uniform_mean_risk,
    gaussian_location_upper_asymptotic_constant,
    gaussian_location_upper_normal_cdf,
    gaussian_location_upper_normal_pdf,
    gaussian_location_upper_shape_parameter,
    gaussian_location_upper_shape_root,
    gaussian_minimal_log_derivative_upper_numerator,
    gaussian_minimal_log_derivative_upper_numerator_derivative,
    gaussian_minimal_proof_gap_ratio_lower_bound,
    gaussian_minimal_proof_gap_ratio_lower_bound_log_derivative_upper,
    gaussian_minimal_proof_gap_ratio_lower_bound_right_endpoint,
    gaussian_minimal_proof_gap,
    gaussian_minimal_proof_gap_global_infimum,
    gaussian_minimal_proof_gap_infimum_for_H,
    gaussian_minimal_proof_gap_ratio,
    gaussian_ramp_piecewise_left_limit_lower_bound,
    gaussian_ramp_piecewise_right_endpoint_lower_bound,
    gaussian_ramp_piecewise_threshold_lower_bound,
    gaussian_ramp_piecewise_threshold_power,
    gaussian_ramp_piecewise_lower_bound,
    gaussian_ramp_piecewise_lower_bound_log_second_derivative,
    gaussian_ramp_piecewise_threshold_H,
    gaussian_ramp_proof_gap,
    gaussian_ramp_proof_gap_ratio_lower_bound,
    gaussian_ramp_proof_gap_global_infimum,
    gaussian_ramp_proof_gap_infimum_for_H,
    gaussian_ramp_proof_gap_ratio,
    komatsu_gaussian_tail_upper,
    run_sharp_family_audit,
    supplement_candidate_constant,
    supplement_gap_threshold,
    supplement_proof_gap,
    supplement_proof_gap_global_infimum,
    supplement_proof_gap_infimum_for_H,
    supplement_proof_gap_ratio,
    uniform_window_staleness_constant,
)


class SharpFamilyAuditTest(unittest.TestCase):
    def test_finite_n_staleness_constant_converges_to_closed_form_limit(self) -> None:
        for H in (0.25, 0.5, 0.75, 1.0):
            finite_n = uniform_window_staleness_constant(H, 4096)
            self.assertAlmostEqual(
                finite_n,
                asymptotic_staleness_constant(H),
                delta=8e-4,
            )

    def test_dirac_staleness_matches_constant_formula_exactly(self) -> None:
        zeta = 0.7
        H = 0.75
        n = 64
        expected = zeta * uniform_window_staleness_constant(H, n) * (n**H)
        self.assertAlmostEqual(
            dirac_uniform_window_staleness(zeta, H, n),
            expected,
            places=12,
        )

    def test_numeric_lower_bound_tracks_current_asymptotic_constant(self) -> None:
        result = run_sharp_family_audit(
            SharpFamilyAuditConfig(
                H_values=(0.5, 0.75, 1.0),
                sigma_zeta_ratios=(10_000.0,),
                n_values=(64, 256, 1024),
                max_multiplier=4.0,
            )
        )

        for row in result.lower_bound_rows:
            H = float(row["H"])
            self.assertAlmostEqual(
                float(row["normalized_best"]),
                Hölder_asymptotic_constant(H),
                delta=1.5e-3,
            )

    def test_supplement_candidate_constant_is_not_the_current_witness_constant(
        self,
    ) -> None:
        for H in (0.5, 0.75, 1.0):
            self.assertGreater(
                supplement_candidate_constant(H),
                2.5 * Hölder_asymptotic_constant(H),
            )

    def test_gaussian_location_upper_shape_root_solves_stationary_equation(
        self,
    ) -> None:
        for H in (0.35, 0.5, 0.75, 1.0):
            root = gaussian_location_upper_shape_root(H)
            residual = H * root * (
                2.0 * gaussian_location_upper_normal_cdf(root) - 1.0
            ) - gaussian_location_upper_normal_pdf(root)
            self.assertAlmostEqual(residual, 0.0, places=12)

    def test_gaussian_location_upper_constant_matches_discrete_optimization(
        self,
    ) -> None:
        for H in (0.35, 0.5, 0.75, 1.0):
            for ratio in (1_000.0, 10_000.0):
                sigma = ratio
                zeta = 1.0
                sigma_power = 2.0 * H / (2.0 * H + 1.0)
                zeta_power = 1.0 / (2.0 * H + 1.0)
                predicted_h = gaussian_location_upper_shape_parameter(H) * ratio ** (
                    2.0 / (2.0 * H + 1.0)
                )
                h_min = max(1, int(predicted_h / 3.0))
                h_max = int(3.0 * predicted_h) + 50
                partial_sum = sum((j**H) for j in range(h_min))
                best_n = h_min
                best_bias = zeta * partial_sum / h_min
                best_risk = gaussian_location_expected_absolute_error(
                    best_bias, sigma / math.sqrt(h_min)
                )
                for n in range(h_min + 1, h_max + 1):
                    partial_sum += (n - 1) ** H
                    bias = zeta * partial_sum / n
                    risk = gaussian_location_expected_absolute_error(
                        bias, sigma / math.sqrt(n)
                    )
                    if risk < best_risk:
                        best_n = n
                        best_risk = risk
                normalized_risk = best_risk / (sigma**sigma_power * zeta**zeta_power)
                self.assertAlmostEqual(
                    normalized_risk,
                    gaussian_location_upper_asymptotic_constant(H),
                    delta=3e-3,
                )
                normalized_h = best_n / (ratio ** (2.0 / (2.0 * H + 1.0)))
                self.assertAlmostEqual(
                    normalized_h,
                    gaussian_location_upper_shape_parameter(H),
                    delta=0.03,
                )

    def test_supplement_gap_threshold_is_a_root_of_ratio_minus_one(self) -> None:
        threshold = supplement_gap_threshold()
        self.assertAlmostEqual(supplement_proof_gap_ratio(threshold), 1.0, places=12)

    def test_supplement_gap_infimum_for_fixed_H_matches_boundary_formula(self) -> None:
        for H in (0.25, 0.5, 0.75, 1.0):
            predicted = supplement_proof_gap_infimum_for_H(H)
            threshold = supplement_gap_threshold()
            numeric = (
                supplement_proof_gap(1e-8, H)
                if H < threshold
                else supplement_proof_gap(1e8, H)
            )
            self.assertAlmostEqual(numeric, predicted, delta=2e-3)

    def test_supplement_gap_infimum_equals_two_below_threshold(self) -> None:
        threshold = supplement_gap_threshold()
        for H in (0.1, 0.25, threshold * 0.99):
            self.assertAlmostEqual(
                supplement_proof_gap_infimum_for_H(H), 2.0, places=10
            )

    def test_supplement_gap_infimum_drops_below_two_above_threshold(self) -> None:
        threshold = supplement_gap_threshold()
        for H in (max(0.5, threshold * 1.01), 0.75, 1.0):
            self.assertLess(supplement_proof_gap_infimum_for_H(H), 2.0)

    def test_supplement_gap_global_infimum_is_attained_at_H_equals_one_limit(
        self,
    ) -> None:
        self.assertAlmostEqual(
            supplement_proof_gap_global_infimum(),
            supplement_proof_gap_infimum_for_H(1.0),
            places=12,
        )
        self.assertAlmostEqual(
            supplement_proof_gap_global_infimum(),
            2.0 * supplement_proof_gap_ratio(1.0),
            places=12,
        )

    def test_refined_ramp_gap_ratio_stays_above_one(self) -> None:
        for H in [k / 200.0 for k in range(1, 201)]:
            self.assertGreater(gaussian_ramp_proof_gap_ratio(H), 1.0)

    def test_refined_ramp_gap_ratio_lower_bound_sits_below_true_ratio(self) -> None:
        for H in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
            self.assertLessEqual(
                gaussian_ramp_proof_gap_ratio_lower_bound(H),
                gaussian_ramp_proof_gap_ratio(H),
            )

    def test_ramp_piecewise_lower_bound_stays_above_one(self) -> None:
        for H in [k / 2000.0 for k in range(1, 2001)]:
            self.assertGreater(gaussian_ramp_piecewise_lower_bound(H), 1.0)

    def test_ramp_piecewise_lower_bound_log_second_derivative_is_negative(self) -> None:
        for H in [k / 2000.0 for k in range(1, 2001)]:
            self.assertLess(
                gaussian_ramp_piecewise_lower_bound_log_second_derivative(H), 0.0
            )

    def test_ramp_piecewise_threshold_matches_expected_endpoint_checks(self) -> None:
        threshold = gaussian_ramp_piecewise_threshold_H()
        self.assertGreater(gaussian_ramp_piecewise_lower_bound(threshold * 0.999), 1.0)
        self.assertGreater(gaussian_ramp_piecewise_lower_bound(threshold * 1.001), 1.0)

    def test_ramp_piecewise_endpoint_lower_bounds_stay_above_one(self) -> None:
        self.assertAlmostEqual(gaussian_ramp_piecewise_left_limit_lower_bound(), 2.0)
        self.assertGreater(gaussian_ramp_piecewise_threshold_lower_bound(), 1.0)
        self.assertGreater(gaussian_ramp_piecewise_right_endpoint_lower_bound(), 1.0)

    def test_ramp_piecewise_threshold_power_matches_threshold_H(self) -> None:
        self.assertAlmostEqual(
            gaussian_ramp_piecewise_threshold_power(),
            1.0 - 2.0 / math.pi,
            places=12,
        )
        threshold_H = gaussian_ramp_piecewise_threshold_H()
        threshold_power = gaussian_ramp_piecewise_threshold_power()
        self.assertAlmostEqual(
            2.0 * threshold_H / (2.0 * threshold_H + 1.0),
            threshold_power,
            places=12,
        )

    def test_refined_minimal_gap_ratio_stays_above_one(self) -> None:
        for H in [k / 2000.0 for k in range(1, 2001)]:
            self.assertGreater(gaussian_minimal_proof_gap_ratio(H), 1.0)

    def test_refined_ramp_gap_infimum_is_two_for_fixed_H(self) -> None:
        for H in (0.1, 0.25, 0.5, 0.75, 1.0):
            self.assertAlmostEqual(
                gaussian_ramp_proof_gap_infimum_for_H(H), 2.0, places=12
            )
            self.assertAlmostEqual(gaussian_ramp_proof_gap(1e-8, H), 2.0, delta=2e-6)

    def test_refined_minimal_gap_infimum_is_two_for_fixed_H(self) -> None:
        for H in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
            self.assertAlmostEqual(
                gaussian_minimal_proof_gap_infimum_for_H(H), 2.0, places=12
            )
            self.assertAlmostEqual(gaussian_minimal_proof_gap(1e-8, H), 2.0, delta=1e-5)

    def test_refined_gap_global_infima_equal_two(self) -> None:
        self.assertAlmostEqual(gaussian_ramp_proof_gap_global_infimum(), 2.0, places=12)
        self.assertAlmostEqual(
            gaussian_minimal_proof_gap_global_infimum(), 2.0, places=12
        )

    def test_komatsu_upper_bound_dominates_gaussian_tail(self) -> None:
        import math

        for x in (0.0, 0.1, 0.3, 0.5, 0.75, 1.0, 2.0):
            exact_tail = 0.5 * (1.0 - math.erf(x / math.sqrt(2.0)))
            self.assertGreaterEqual(komatsu_gaussian_tail_upper(x), exact_tail)

    def test_minimal_gap_ratio_lower_bound_sits_below_true_ratio(self) -> None:
        for H in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
            self.assertLessEqual(
                gaussian_minimal_proof_gap_ratio_lower_bound(H),
                gaussian_minimal_proof_gap_ratio(H),
            )

    def test_minimal_gap_ratio_lower_bound_stays_above_one(self) -> None:
        for H in [k / 2000.0 for k in range(1, 2001)]:
            self.assertGreater(gaussian_minimal_proof_gap_ratio_lower_bound(H), 1.0)
        self.assertGreater(
            gaussian_minimal_proof_gap_ratio_lower_bound_right_endpoint(), 1.0
        )

    def test_minimal_gap_ratio_lower_bound_log_derivative_upper_is_negative(
        self,
    ) -> None:
        for H in [k / 2000.0 for k in range(1, 2001)]:
            self.assertLess(
                gaussian_minimal_proof_gap_ratio_lower_bound_log_derivative_upper(H),
                0.0,
            )

    def test_minimal_log_derivative_upper_numerator_is_negative(self) -> None:
        for H in [k / 2000.0 for k in range(1, 2001)]:
            self.assertLess(gaussian_minimal_log_derivative_upper_numerator(H), 0.0)

    def test_minimal_log_derivative_upper_numerator_derivative_is_negative(
        self,
    ) -> None:
        for H in [k / 2000.0 for k in range(1, 2001)]:
            self.assertLess(
                gaussian_minimal_log_derivative_upper_numerator_derivative(H),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
