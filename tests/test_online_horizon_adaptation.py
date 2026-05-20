from __future__ import annotations

import unittest

import numpy as np

from useful_memory_horizon.online_horizon_adaptation import (
    OnlineAdaptationConfig,
    activity_window_from_proxy,
    estimate_aggregated_roughness,
    estimate_local_activity,
    estimate_local_roughness,
    phase_profile,
    run_online_horizon_adaptation_experiment,
)


class OnlineHorizonAdaptationTest(unittest.TestCase):
    def test_phase_profile_realizes_holder_phase_law_within_each_phase(self) -> None:
        config = OnlineAdaptationConfig(
            phase_lengths=(60,),
            holder_exponents=(0.5,),
            roughness_scales=(0.2,),
            phase_signs=(1.0,),
        )
        latent_mean, _, _ = phase_profile(config)
        self.assertAlmostEqual(
            latent_mean[25] - latent_mean[9], 0.2 * ((25**0.5) - (9**0.5)), places=12
        )

    def test_local_roughness_estimator_recovers_noiseless_holder_parameters(
        self,
    ) -> None:
        T = 220
        H = 0.5
        zeta = 0.08
        config = OnlineAdaptationConfig(
            phase_lengths=(T,),
            holder_exponents=(H,),
            roughness_scales=(zeta,),
            phase_signs=(1.0,),
            observation_scale=0.0,
            roughness_block_size=1,
            seed=0,
        )
        latent_mean = np.asarray(
            [-zeta * ((T - 1) - t) ** H for t in range(T)], dtype=float
        )
        estimate = estimate_local_roughness(latent_mean, T - 1, config)
        self.assertAlmostEqual(estimate.holder_exponent, 0.5, delta=0.08)
        self.assertAlmostEqual(estimate.roughness_scale, 0.08, delta=0.02)

    def test_activity_window_shortens_as_proxy_increases(self) -> None:
        config = OnlineAdaptationConfig()
        low = activity_window_from_proxy(0.001, config)
        high = activity_window_from_proxy(0.05, config)
        self.assertLessEqual(high, low)

    def test_local_activity_estimator_detects_step_up_in_recent_motion(self) -> None:
        config = OnlineAdaptationConfig(roughness_block_size=8)
        values = np.zeros(80, dtype=float)
        values[-16:-8] = np.linspace(0.0, 0.1, 8)
        values[-8:] = np.linspace(0.2, 0.6, 8)
        self.assertGreater(estimate_local_activity(values, 79, config), 0.0)

    def test_hybrid_policy_stays_close_to_oracle_and_beats_best_static(self) -> None:
        config = OnlineAdaptationConfig(seed=7)
        result = run_online_horizon_adaptation_experiment(config)
        self.assertLessEqual(result.mean_adaptive_error, 1.4 * result.mean_oracle_error)
        self.assertLess(result.mean_adaptive_error, result.mean_best_static_error)
        self.assertGreater(np.std(result.adaptive_window.astype(float)), 10.0)
        self.assertLess(result.mean_activity_error, result.mean_best_static_error)

    def test_structural_policy_improves_on_plugin_and_activity_baselines(self) -> None:
        config = OnlineAdaptationConfig(seed=7)
        result = run_online_horizon_adaptation_experiment(config)
        self.assertLess(result.mean_structural_error, result.mean_plugin_error)
        self.assertLess(result.mean_structural_error, result.mean_activity_error)
        self.assertLess(result.mean_structural_error, result.mean_best_static_error)

    def test_aggregated_roughness_estimator_stays_bounded_on_noisy_stream(self) -> None:
        config = OnlineAdaptationConfig(seed=7)
        latent_mean, _, _ = phase_profile(config)
        rng = np.random.default_rng(config.seed)
        observations = latent_mean + rng.normal(
            scale=config.observation_scale, size=latent_mean.size
        )
        estimate = estimate_aggregated_roughness(
            observations, latent_mean.size - 1, config
        )
        self.assertGreaterEqual(estimate.holder_exponent, config.holder_clip[0])
        self.assertLessEqual(estimate.holder_exponent, config.holder_clip[1])
        self.assertGreaterEqual(estimate.roughness_scale, config.roughness_floor)

    def test_hysteresis_limit_caps_window_index_jumps(self) -> None:
        config = OnlineAdaptationConfig(seed=7, max_window_index_jump=1)
        result = run_online_horizon_adaptation_experiment(config)
        index_map = {window: idx for idx, window in enumerate(config.candidate_windows)}
        jumps = [
            abs(index_map[int(curr)] - index_map[int(prev)])
            for prev, curr in zip(
                result.adaptive_window[:-1], result.adaptive_window[1:], strict=True
            )
        ]
        self.assertLessEqual(max(jumps, default=0), 1)


if __name__ == "__main__":
    unittest.main()
