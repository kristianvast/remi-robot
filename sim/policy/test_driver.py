"""PyTorch (L4, evaluate.py) vs ONNX (Mac CPU) on the same recorded observations: within 1 PWM, >= 10 Hz.

  sim/policy/.venv/bin/python sim/policy/test_driver.py
"""

from pathlib import Path

from run_driver import replay

HERE = Path(__file__).resolve().parent


def test_onnx_matches_pytorch() -> None:
    frames, hz, median_ms, p95_ms, error = replay(HERE / "driver.onnx", HERE / "replay.npz")
    print(f"frames={frames} hz={hz:.1f} median_ms={median_ms:.3f} p95_ms={p95_ms:.3f} max_pwm_difference={error}")
    assert frames > 0
    assert error <= 1
    assert hz >= 10


if __name__ == "__main__":
    test_onnx_matches_pytorch()
    print("PASS")
