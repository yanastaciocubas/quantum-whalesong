#!/usr/bin/env python3
"""
quantum_whalesong.py

Listening to quantum waves through the voices of the ocean.

Features:
- Runs small quantum circuits (Qiskit or Cirq if available; synthetic fallback)
- Extracts statevector (amplitudes + phases)
- Maps amplitudes -> pitch, probabilities -> volume, phases -> stereo panning
- Detects entanglement between qubit pairs and applies entanglement harmonics
- Synthesizes a whale-like stereo WAV using numpy + wave (no external audio libs required)
- Optional MIDI export (mido) and JSON export for visuals
- Random-walk "quantum remix" generator
- CLI with presets: bell, qft, grover, superposition, remix

Usage examples:
    pip install qiskit numpy              # Qiskit route
    python quantum_whalesong.py --preset bell
    python quantum_whalesong.py --preset qft --n 3 --outfile demo.wav --export-json
    python quantum_whalesong.py --preset remix --remix-steps 20 --outfile remix.wav
"""

from __future__ import annotations
import argparse
import math
import json
import wave
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import time
import random

# -------------------------
# Optional extras (mido/midi)
# -------------------------
try:
    import mido
    from mido import Message, MidiFile, MidiTrack
    MIDO_AVAILABLE = True
except Exception:
    MIDO_AVAILABLE = False

# -------------------------
# Quantum backends: try Qiskit then Cirq
# -------------------------
QISKIT_AVAILABLE = False
CIRQ_AVAILABLE = False
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except Exception:
    try:
        import cirq
        CIRQ_AVAILABLE = True
    except Exception:
        QISKIT_AVAILABLE = False
        CIRQ_AVAILABLE = False

# -------------------------
# Output directory
# -------------------------
EXPORTS = Path("exports")
EXPORTS.mkdir(exist_ok=True)

# -------------------------
# Utilities: small helpers
# -------------------------
def ensure_int(x, default=1):
    try:
        return int(x)
    except Exception:
        return default

def log(msg: str):
    print(msg, file=sys.stderr)

# -------------------------
# Quantum circuit builders
# -------------------------
def build_qc_qiskit(preset: str, n_qubits: int) -> "QuantumCircuit":
    """Build a Qiskit QuantumCircuit for given preset."""
    qc = QuantumCircuit(max(1, n_qubits) if preset != "bell" else 2)
    if preset == "bell":
        qc = QuantumCircuit(2)
        qc.h(0); qc.cx(0, 1)
    elif preset == "superposition":
        for q in range(n_qubits):
            qc.h(q)
    elif preset == "qft":
        for j in range(n_qubits):
            qc.h(j)
            for k in range(j + 1, n_qubits):
                angle = math.pi / (2 ** (k - j))
                qc.cp(angle, k, j)  # control k -> target j (common convention)
        for i in range(n_qubits // 2):
            qc.swap(i, n_qubits - i - 1)
    elif preset == "grover":
        # Simple grover-like single oracle marking |11..1>
        for q in range(n_qubits):
            qc.h(q)
        # oracle: flip phase of |11..1>
        if n_qubits == 1:
            qc.z(0)
        else:
            target = n_qubits - 1
            qc.h(target)
            try:
                qc.mcx(list(range(0, target)), target)
            except Exception:
                # fallback to chain of CZs (less ideal but safer)
                for c in range(0, target):
                    qc.cz(c, target)
            qc.h(target)
        # diffusion
        for q in range(n_qubits):
            qc.h(q); qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            target = n_qubits - 1
            qc.h(target)
            try:
                qc.mcx(list(range(0, target)), target)
            except Exception:
                for c in range(0, target):
                    qc.cz(c, target)
            qc.h(target)
        for q in range(n_qubits):
            qc.x(q); qc.h(q)
    else:
        raise ValueError("Unknown preset")
    return qc

def build_qc_cirq(preset: str, n_qubits: int) -> "cirq.Circuit":
    """Build a Cirq circuit for the preset."""
    import cirq
    q_count = max(1, n_qubits) if preset != "bell" else 2
    qs = [cirq.LineQubit(i) for i in range(q_count)]
    ops = []
    if preset == "bell":
        ops = [cirq.H(qs[0]), cirq.CNOT(qs[0], qs[1])]
    elif preset == "superposition":
        for q in qs:
            ops.append(cirq.H(q))
    elif preset == "qft":
        # simple QFT
        n = q_count
        for j in range(n):
            ops.append(cirq.H(qs[j]))
            for k in range(j + 1, n):
                angle = math.pi / (2 ** (k - j))
                ops.append(cirq.CZ(qs[k], qs[j]) ** (angle / math.pi))
        for i in range(n // 2):
            ops.append(cirq.SWAP(qs[i], qs[n - i - 1]))
    elif preset == "grover":
        n = q_count
        for q in qs:
            ops.append(cirq.H(q))
        # oracle: flip |11..1>
        if n == 1:
            ops.append(cirq.Z(qs[0]))
        else:
            # approximate with multi-controlled Z via ancilla-free decomposition is involved;
            # use chain of CZ (approx)
            target = qs[-1]
            for c in qs[:-1]:
                ops.append(cirq.CZ(c, target))
        # diffusion
        for q in qs:
            ops.append(cirq.H(q)); ops.append(cirq.X(q))
        if n == 1:
            ops.append(cirq.Z(qs[0]))
        else:
            target = qs[-1]
            for c in qs[:-1]:
                ops.append(cirq.CZ(c, target))
        for q in qs:
            ops.append(cirq.X(q)); ops.append(cirq.H(q))
    return cirq.Circuit(ops)

# -------------------------
# Statevector extraction
# -------------------------
def get_statevector_from_qiskit(qc) -> np.ndarray:
    # Use Statevector.from_instruction for stability
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(qc)
    return np.array(sv.data, dtype=np.complex128)

def get_statevector_from_cirq(circuit) -> np.ndarray:
    import cirq
    sim = cirq.Simulator()
    res = sim.simulate(circuit)
    sv = res.state_vector()
    return np.array(sv, dtype=np.complex128)

# -------------------------
# Entanglement detection
# -------------------------
def reduced_density_matrix(state: np.ndarray, total_qubits: int, keep: List[int]) -> np.ndarray:
    """
    Compute reduced density matrix (partial trace) keeping qubits in `keep` list.
    state: complex vector length 2**total_qubits
    keep: list of qubit indices to keep (0 is least significant or most? here we use conventional ordering: index 0 is qubit 0)
    Returns a matrix of size 2**len(keep).
    """
    # reshape state to tensor of shape (2,)*n
    state_tensor = state.reshape([2] * total_qubits)
    # axes order: keep indices first, then traced out
    keep = list(keep)
    trace_out = [i for i in range(total_qubits) if i not in keep]
    # transpose so keep axes come first
    perm = keep + trace_out
    transposed = np.transpose(state_tensor, perm)
    k = len(keep)
    keep_dim = 2 ** k
    trace_dim = 2 ** (total_qubits - k)
    # reshape to (keep_dim, trace_dim)
    mat = transposed.reshape((keep_dim, trace_dim))
    # density matrix = mat @ mat.conj().T? Actually we need to compute partial trace:
    # Construct density operator and trace out trace_dim indices:
    # easier: build full density operator then trace
    full_rho = np.outer(state, np.conjugate(state)).reshape([2] * (2 * total_qubits))
    # Permute axes so keep indices appear first in both bras and kets
    keep_axes = keep
    keep_axes_pairs = keep_axes + [i + total_qubits for i in keep_axes]
    trace_axes = trace_out
    trace_axes_pairs = trace_axes + [i + total_qubits for i in trace_axes]
    perm2 = keep + trace_out + [i + total_qubits for i in keep] + [i + total_qubits for i in trace_out]
    rho_perm = np.transpose(full_rho, perm2)
    # reshape to (keep_dim, trace_dim, keep_dim, trace_dim)
    rho_rs = rho_perm.reshape((keep_dim, trace_dim, keep_dim, trace_dim))
    # partial trace over trace_dim
    reduced = np.einsum('ijkt->ik', rho_rs)
    return reduced

def is_entangled_pair(state: np.ndarray, total_qubits: int, a: int, b: int, tol=0.9999) -> bool:
    """
    Rough test: compute purity of reduced density matrix for the pair.
    For a pure global state, if the reduced state of the pair is mixed (purity < 1) then it is entangled with rest;
    but to test entanglement between a and b specifically, check if reduced two-qubit state is separable is hard.
    Here we'll use a heuristic: compute reduced density of pair; if its purity < tol (i.e., not pure) then
    it is entangled with other qubits; additionally check concurrence for 2-qubit case.
    """
    k = [a, b]
    red = reduced_density_matrix(state, total_qubits, k)
    # purity
    pur = np.real(np.trace(red @ red))
    if total_qubits == 2:
        # for 2-qubit pure state, compute concurrence
        psi = state
        # reshape to 2x2
        try:
            psi = psi.reshape((2, 2))
            # concurrence for pure psi = 2 * |det(psi)|
            det = np.linalg.det(psi)
            conc = 2 * abs(det)
            return conc > 1e-6
        except Exception:
            return pur < tol
    return pur < tol

# -------------------------
# Sonification mapping: amplitude->freq, prob->volume, phase->pan
# -------------------------
def map_index_to_freq(index: int, n_states: int, f_min=30.0, f_max=420.0, scale="log") -> float:
    if n_states <= 1:
        return (f_min + f_max) / 2.0
    if scale == "log":
        frac = index / (n_states - 1)
        return f_min * (f_max / f_min) ** frac
    else:
        return f_min + (f_max - f_min) * (index / (n_states - 1))

def state_to_voice_params(state: np.ndarray, qcount: int, f_min=30.0, f_max=300.0) -> List[Dict]:
    amps = np.abs(state)
    probs = amps ** 2
    maxp = probs.max() if probs.size else 0.0
    if maxp > 0:
        probs = probs / maxp
    else:
        probs = np.zeros_like(probs)
    n = len(state)
    voices = []
    for i, (a, p) in enumerate(zip(amps, probs)):
        freq = map_index_to_freq(i, n, f_min=f_min, f_max=f_max, scale="log")
        phase = math.atan2(state[i].imag, state[i].real)
        voices.append({
            "index": i,
            "amp": float(a),
            "prob": float(p),
            "phase": float(phase),
            "freq": float(freq),
            "basis": format(i, f"0{qcount}b")
        })
    # sort by prob desc (stronger first)
    voices.sort(key=lambda v: v["prob"], reverse=True)
    return voices

# -------------------------
# Synthesis primitives (whale-like)
# -------------------------
def constant_power_pan(mono: np.ndarray, pan: float) -> np.ndarray:
    pan = float(max(-1.0, min(1.0, pan)))
    theta = (pan + 1.0) * (math.pi / 4.0)
    left = mono * math.cos(theta)
    right = mono * math.sin(theta)
    return np.vstack([left, right]).T

def synth_whale_voice(freq: float, prob: float, phase: float,
                      duration: float = 4.0, sr: int = 44100,
                      vibrato_rate=0.2, vibrato_depth=0.02, richness=0.8) -> np.ndarray:
    n = int(duration * sr)
    t = np.linspace(0.0, duration, n, endpoint=False)
    # glide
    glide = 1.0 + 0.08 * (1.0 - prob)
    freq_t = np.linspace(freq, freq * glide, n)
    # vibrato
    vibr = vibrato_depth * np.sin(2 * math.pi * vibrato_rate * t)
    phase_arr = 2 * math.pi * (np.cumsum(freq_t / sr) + vibr) + phase
    fund = np.sin(phase_arr)
    second = 0.25 * np.sin(2 * phase_arr)
    sub = 0.5 * np.sin(0.5 * phase_arr) * prob
    # noise
    rng = np.random.default_rng(seed=int((freq * 1000) % (2**31)))
    noise = rng.standard_normal(n) * (0.12 * (1.0 - prob))
    # smoothing noise
    ksize = max(1, int(sr * 0.01))
    if ksize > 1:
        kernel = np.ones(ksize) / ksize
        noise = np.convolve(noise, kernel, mode="same")
    # envelope
    attack = int(0.06 * n)
    release = int(0.18 * n)
    env = np.ones(n)
    if attack > 0:
        env[:attack] = np.linspace(0.0, 1.0, attack)
    if release > 0:
        env[-release:] = np.linspace(1.0, 0.0, release)
    mono = (0.9 * fund + richness * second + sub) * env
    mono += noise * env
    mono *= (0.6 * prob + 0.05)
    pan = float(max(-1.0, min(1.0, phase / math.pi)))
    stereo = constant_power_pan(mono, pan)
    return stereo.astype(np.float32)

def mix_and_normalize(tracks: List[np.ndarray]) -> np.ndarray:
    if not tracks:
        return np.zeros((0, 2), dtype=np.float32)
    maxlen = max(t.shape[0] for t in tracks)
    mix = np.zeros((maxlen, 2), dtype=np.float32)
    for t in tracks:
        L = t.shape[0]
        mix[:L, :] += t
    peak = np.max(np.abs(mix)) if np.max(np.abs(mix)) > 0 else 1.0
    mix = mix / peak * 0.95
    return mix

def simple_reverb(audio: np.ndarray, sr: int = 44100, rt=1.0) -> np.ndarray:
    n_kernel = max(2, int(rt * sr))
    times = np.arange(n_kernel) / sr
    decay = np.exp(-times / rt).astype(np.float32)
    left = np.convolve(audio[:, 0], decay, mode='full')[:audio.shape[0]]
    right = np.convolve(audio[:, 1], decay, mode='full')[:audio.shape[0]]
    out = np.vstack([left, right]).T
    peak = np.max(np.abs(out)) if np.max(np.abs(out)) > 0 else 1.0
    out = out / peak * 0.95
    return out

def write_wav(path: Path, audio: np.ndarray, sr: int = 44100):
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("Audio must be stereo (n,2)")
    audio_i16 = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(audio_i16.tobytes())

# -------------------------
# Entanglement-guided harmonics
# -------------------------
def apply_entanglement_resonance(voices: List[Dict], state: np.ndarray, qcount: int) -> List[Dict]:
    """
    For each pair of qubits, if entangled then modify voice params so entangled basis states
    create harmonic overlays: if states i and j correspond to same pair correlated indices,
    we blend frequencies (e.g., add small sideband).
    Here we do a simple rule: identify entangled pairs and for voices whose basis strings show
    correlated bits on that pair, add a harmonic modulation factor.
    """
    n = len(state)
    # check each pair
    for a in range(qcount):
        for b in range(a+1, qcount):
            try:
                ent = is_entangled_pair(state, qcount, a, b)
            except Exception:
                ent = False
            if ent:
                # apply resonance: for voices whose basis bits at a,b are both '1' or correlated, boost richness
                for v in voices:
                    basis = v.get("basis", "")
                    if len(basis) == qcount and basis[a] == basis[b]:
                        # boost richness and prob slightly for audible effect
                        v["richness"] = v.get("richness", 0.8) * 1.25
                        v["prob"] = min(1.0, v.get("prob", 0.0) + 0.08)
    return voices

# -------------------------
# Random-walk quantum remix (phase/freq tweaks)
# -------------------------
def random_walk_remix(state: np.ndarray, steps: int = 16, seed: Optional[int] = None) -> np.ndarray:
    rng = random.Random(seed)
    s = state.copy()
    n = len(s)
    for step in range(steps):
        i = rng.randrange(n)
        j = rng.randrange(n)
        # swap small phase component between i and j
        phi_i = math.atan2(s[i].imag, s[i].real)
        phi_j = math.atan2(s[j].imag, s[j].real)
        delta = (phi_j - phi_i) * 0.08
        # apply small rotation
        s[i] *= complex(math.cos(delta), math.sin(delta))
        s[j] *= complex(math.cos(-delta), math.sin(-delta))
        # small amplitude jitter conserving norm
        ai = abs(s[i]); aj = abs(s[j])
        if ai + aj > 0:
            eps = 0.01 * (rng.random() - 0.5)
            s[i] *= (1.0 + eps)
            s[j] *= (1.0 - eps)
    # renormalize
    norm = np.linalg.norm(s)
    if norm > 0:
        s = s / norm
    return s

# -------------------------
# MIDI export (optional)
# -------------------------
def export_midi(notes: List[Tuple[int,int]], path: Path, tempo: int = 80, hold_beats: int = 4):
    if not MIDO_AVAILABLE:
        log("mido not installed; skipping MIDI export")
        return
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
    hold = mid.ticks_per_beat * hold_beats
    # note_on all
    for i, (note, vel) in enumerate(notes):
        track.append(Message('note_on', note=note, velocity=vel, time=0 if i>0 else 0))
    # off after hold
    first = True
    for note, vel in notes:
        if first:
            track.append(Message('note_off', note=note, velocity=0, time=hold))
            first = False
        else:
            track.append(Message('note_off', note=note, velocity=0, time=0))
    mid.save(path)
    log(f"MIDI saved to {path}")

# -------------------------
# High-level pipeline
# -------------------------
def pipeline(preset: str = "bell",
             n_qubits: int = 3,
             outfile: str = "whalesong.wav",
             export_json: bool = False,
             export_midi_flag: bool = False,
             remix_steps: int = 0,
             duration: float = 4.0,
             sr: int = 44100):
    # Build or simulate circuit
    if QISKIT_AVAILABLE:
        qc = build_qc_qiskit(preset, n_qubits)
        state = get_statevector_from_qiskit(qc)
    elif CIRQ_AVAILABLE:
        qc = build_qc_cirq(preset, n_qubits)
        state = get_statevector_from_cirq(qc)
    else:
        # synthetic fallback: bell or uniform superposition
        log("No Qiskit or Cirq found — using synthetic state for demo.")
        if preset == "bell":
            state = np.array([1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], dtype=np.complex128)
            n_qubits = 2
        else:
            dim = 2 ** max(1, n_qubits)
            state = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)

    if remix_steps > 0:
        state = random_walk_remix(state, steps=remix_steps, seed=int(time.time() % 1e9))

    # voice params
    voices = state_to_voice_params(state, qcount=int(round(math.log2(len(state)))) if len(state)>1 else 1,
                                   f_min=35.0, f_max=420.0)

    # entanglement resonance
    try:
        voices = apply_entanglement_resonance(voices, state, qcount=int(round(math.log2(len(state)))) if len(state)>1 else 1)
    except Exception:
        pass

    # synthesize voices
    tracks = []
    notes_for_midi: List[Tuple[int,int]] = []
    for v in voices:
        dur = max(0.6, duration * (0.9 + 0.6 * v["prob"]))
        richness = v.get("richness", 0.8)
        stereo = synth_whale_voice(freq=v["freq"], prob=v["prob"], phase=v["phase"],
                                   duration=dur, sr=sr, vibrato_rate=0.18 + 0.08*(1.0-v["prob"]),
                                   vibrato_depth=0.01 + 0.02*v["prob"], richness=richness)
        tracks.append(stereo)
        # approximate MIDI note from frequency for midi export (A440=69)
        midi_note = int(np.clip(69 + 12 * math.log2(max(1.0, v["freq"]) / 440.0), 0, 127))
        vel = int(np.clip(30 + v["prob"] * 90, 0, 127))
        notes_for_midi.append((midi_note, vel))

    mix = mix_and_normalize(tracks)
    mix = simple_reverb(mix, sr=sr, rt=1.1)
    out_path = EXPORTS / outfile
    write_wav(out_path, mix, sr=sr)
    log(f"WAV written to: {out_path}")

    if export_json:
        js = {
            "preset": preset,
            "n_qubits": int(round(math.log2(len(state)))) if len(state)>1 else 1,
            "amplitudes": np.abs(state).tolist(),
            "phases": np.angle(state).tolist(),
            "voices": voices
        }
        json_path = EXPORTS / (Path(outfile).stem + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(js, f, indent=2)
        log(f"JSON written to: {json_path}")

    if export_midi_flag and MIDO_AVAILABLE:
        midi_path = EXPORTS / (Path(outfile).stem + ".mid")
        export_midi(notes_for_midi, midi_path, tempo=80, hold_beats=8)

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Quantum Whalesong — listen to quantum waves")
    parser.add_argument("--preset", choices=["bell", "qft", "grover", "superposition", "remix"], default="bell",
                        help="Circuit preset (remix uses random-walk over a base state).")
    parser.add_argument("--n", type=int, default=3, help="Number of qubits for qft/grover/superposition.")
    parser.add_argument("--outfile", type=str, default="whalesong.wav", help="Output WAV filename in exports/")
    parser.add_argument("--export-json", action="store_true", help="Export amplitudes/phases to JSON for visuals.")
    parser.add_argument("--export-midi", action="store_true", help="Export MIDI (if mido installed).")
    parser.add_argument("--remix-steps", type=int, default=0, help="Apply random-walk remix steps to statevector.")
    parser.add_argument("--duration", type=float, default=3.6, help="Base duration per voice in seconds.")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate for WAV.")
    args = parser.parse_args()

    preset = args.preset
    if preset == "remix":
        # use superposition base then remix
        preset = "superposition"
        remix_steps = max(8, args.remix_steps or 16)
    else:
        remix_steps = args.remix_steps

    if preset == "bell":
        n_qubits = 2
    else:
        n_qubits = max(1, args.n)

    pipeline(preset=preset, n_qubits=n_qubits, outfile=args.outfile,
             export_json=args.export_json, export_midi_flag=args.export_midi,
             remix_steps=remix_steps, duration=args.duration, sr=args.sr)

if __name__ == "__main__":
    main()

# =============================================================================
# Developer Notes
# =============================================================================
# This file is designed to be modular and extensible.
#
# To add new quantum circuits:
#     1. Create a new method in QuantumWhalesong (e.g., create_my_algorithm()).
#     2. Use Qiskit gates to define your algorithm.
#     3. Return a QuantumCircuit object.
#
# To extend sound mapping:
#     - Modify amplitudes_to_notes() to experiment with custom musical scales.
#     - Adjust phase-to-panning logic to create spatialized sound effects.
#
# To integrate new visualizations:
#     - Export data to JSON using export_json().
#     - Use Three.js or WebGL to render whales and waves that respond to
#       amplitudes and phases in real time.
#
# To experiment with machine learning:
#     - Use TensorFlow Quantum or PyTorch to analyze statevector patterns.
#     - Generate “quantum remixes” by sampling and modifying amplitude spectra.
#