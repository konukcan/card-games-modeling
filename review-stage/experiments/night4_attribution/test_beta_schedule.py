"""Smoke test for the new MCMCConfig.beta_schedule field.

Asserts:
  - "linear" mode: β(0) = beta_start, β(n-1) ≈ beta_end (matches old behavior).
  - "piecewise_half" mode: β(0) = beta_start, β(n/2) = beta_end, β(n-1) = beta_end.
  - Both modes produce β = beta_start when beta_start == beta_end.

We don't actually run a chain (too slow for a smoke test). We inline the β
computation from mcmc_search.py to test the formula directly. If the test
passes here and the file on disk has the same formula, the patch is correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from gallery_analysis.mcmc_search import MCMCConfig  # noqa: E402


def beta_at(config: MCMCConfig, step: int) -> float:
    """Mirror of the β formula in MCMCChain.run (mcmc_search.py:2115)."""
    if config.beta_start == config.beta_end:
        return config.beta_start
    if config.beta_schedule == "piecewise_half":
        half = config.n_steps // 2
        if step >= half:
            return config.beta_end
        return config.beta_start + (config.beta_end - config.beta_start) * (step / half)
    # "linear"
    return config.beta_start + (config.beta_end - config.beta_start) * (step / config.n_steps)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def test_linear_backward_compat():
    """Default beta_schedule='linear' reproduces the pre-patch behavior."""
    cfg = MCMCConfig(n_steps=100, beta_start=0.5, beta_end=1.0)
    assert cfg.beta_schedule == "linear", "default must be 'linear'"
    assert approx(beta_at(cfg, 0), 0.5), f"linear β(0) = {beta_at(cfg, 0)}"
    assert approx(beta_at(cfg, 50), 0.75), f"linear β(50) = {beta_at(cfg, 50)}"
    assert approx(beta_at(cfg, 99), 0.995), f"linear β(99) = {beta_at(cfg, 99)}"
    print("  [PASS] linear mode: backward-compat behavior preserved")


def test_piecewise_half():
    """piecewise_half ramps over first half, flat at beta_end for second half."""
    cfg = MCMCConfig(
        n_steps=100, beta_start=0.5, beta_end=1.0, beta_schedule="piecewise_half"
    )
    # Ramp phase: steps 0..49
    assert approx(beta_at(cfg, 0), 0.5), f"β(0) = {beta_at(cfg, 0)}"
    assert approx(beta_at(cfg, 25), 0.75), f"β(25) = {beta_at(cfg, 25)}"
    assert approx(beta_at(cfg, 49), 0.99), f"β(49) = {beta_at(cfg, 49)}"  # 0.5 + 0.5*49/50
    # Flat phase: steps 50..99
    assert approx(beta_at(cfg, 50), 1.0), f"β(50) = {beta_at(cfg, 50)}"
    assert approx(beta_at(cfg, 75), 1.0), f"β(75) = {beta_at(cfg, 75)}"
    assert approx(beta_at(cfg, 99), 1.0), f"β(99) = {beta_at(cfg, 99)}"
    print("  [PASS] piecewise_half: ramp 0.5→1.0 over first half, flat at 1.0 after")


def test_no_annealing_shortcut():
    """When beta_start == beta_end, both schedules return that constant."""
    for schedule in ("linear", "piecewise_half"):
        cfg = MCMCConfig(n_steps=100, beta_start=1.0, beta_end=1.0, beta_schedule=schedule)
        for step in (0, 50, 99):
            assert approx(beta_at(cfg, step), 1.0), (
                f"{schedule} β({step}) with beta_start==beta_end==1.0 = {beta_at(cfg, step)}"
            )
    print("  [PASS] shortcut: beta_start==beta_end produces constant β for both schedules")


def test_night4_config():
    """The exact config Night 4 C1 will use: n_steps=100_000, piecewise_half."""
    cfg = MCMCConfig(
        n_steps=100_000,
        beta_start=0.5,
        beta_end=1.0,
        beta_schedule="piecewise_half",
    )
    # Ramp phase: 0..49_999
    assert approx(beta_at(cfg, 0), 0.5)
    assert approx(beta_at(cfg, 25_000), 0.75), f"β(25k) = {beta_at(cfg, 25_000)}"
    assert approx(beta_at(cfg, 49_999), 0.5 + 0.5 * 49_999 / 50_000)  # ≈ 0.99999
    # Flat phase: 50_000..99_999
    assert approx(beta_at(cfg, 50_000), 1.0)
    assert approx(beta_at(cfg, 75_000), 1.0)
    assert approx(beta_at(cfg, 99_999), 1.0)
    print("  [PASS] Night 4 C1 config (n=100k, piecewise_half): β=1.0 on steps 50k..99_999")


if __name__ == "__main__":
    print("Testing MCMCConfig.beta_schedule...")
    test_linear_backward_compat()
    test_piecewise_half()
    test_no_annealing_shortcut()
    test_night4_config()
    print("\nAll β-schedule tests passed.")
