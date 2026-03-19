import math
import numpy as np
import pytest
from pathlib import Path
from quantum_whalesong import (
    map_index_to_freq,
    constant_power_pan,
    is_entangled_pair,
    state_to_voice_params,
    write_wav,
    mix_and_normalize,
)

# -------------------------
# map_index_to_freq
# -------------------------
def test_map_index_to_freq_bounds():
    assert map_index_to_freq(0, 8) == pytest.approx(30.0)
    assert map_index_to_freq(7, 8) == pytest.approx(420.0)

def test_map_index_to_freq_single_state():
    result = map_index_to_freq(0, 1)
    assert result == pytest.approx((30.0 + 420.0) / 2.0)

# -------------------------
# constant_power_pan
# -------------------------
def test_constant_power_pan_center():
    mono = np.ones(100, dtype=np.float32)
    stereo = constant_power_pan(mono, 0.0)
    assert stereo.shape == (100, 2)
    np.testing.assert_allclose(stereo[:, 0], stereo[:, 1], atol=1e-5)

def test_constant_power_pan_clamps():
    mono = np.ones(100, dtype=np.float32)
    stereo_left = constant_power_pan(mono, -1.0)
    stereo_right = constant_power_pan(mono, 1.0)
    assert stereo_left[:, 0].mean() > stereo_left[:, 1].mean()
    assert stereo_right[:, 1].mean() > stereo_right[:, 0].mean()

# -------------------------
# entanglement detection
# -------------------------
def test_bell_state_is_entangled():
    bell = np.array([1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], dtype=np.complex128)
    assert is_entangled_pair(bell, total_qubits=2, a=0, b=1)

def test_product_state_not_entangled():
    # |00> — separable
    product = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    assert not is_entangled_pair(product, total_qubits=2, a=0, b=1)

# -------------------------
# state_to_voice_params
# -------------------------
def test_voice_params_length():
    state = np.array([1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], dtype=np.complex128)
    voices = state_to_voice_params(state, qcount=2)
    assert len(voices) == 4

def test_voice_params_sorted_by_prob():
    state = np.array([1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], dtype=np.complex128)
    voices = state_to_voice_params(state, qcount=2)
    probs = [v["prob"] for v in voices]
    assert probs == sorted(probs, reverse=True)

# -------------------------
# WAV output shape
# -------------------------
def test_write_wav_creates_file(tmp_path):
    audio = np.zeros((44100, 2), dtype=np.float32)
    out = tmp_path / "test.wav"
    write_wav(out, audio, sr=44100)
    assert out.exists()
    assert out.stat().st_size > 0

def test_write_wav_rejects_mono(tmp_path):
    mono = np.zeros((44100,), dtype=np.float32)
    with pytest.raises(ValueError):
        write_wav(tmp_path / "bad.wav", mono)