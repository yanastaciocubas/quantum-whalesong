# 🐋 Quantum Whalesong  
### *Listening to quantum waves through the voices of the ocean.*


---

## Overview  
**Quantum Whalesong** transforms quantum algorithms into whale-inspired melodies and oceanic visuals.  
Each **qubit** becomes a whale.  
Each **amplitude** becomes a tone.  
Each **interference pattern** becomes a ripple across the sea.  

When you run a quantum circuit (via **Qiskit** or **Cirq**), its probability amplitudes are mapped into sound and light.  
The result is an evolving symphony — a conversation between the mathematics of the quantum world and the songs of the deep.

<p align="center">
<i>“The sea, once it casts its spell, holds one in its net of wonder forever.”</i><br>
— <b>Jacques Cousteau</b>, oceanographer and explorer
</p>
---

## How It Works  

| Layer | Description |
|-------|--------------|
| **Quantum Engine** | Executes algorithms such as the Quantum Fourier Transform, Bell States, or Grover’s Search using Qiskit/Cirq. Extracts amplitude and phase data from the resulting statevector. |
| **Sound Mapping** | Converts amplitudes → pitch, phases → stereo panning, magnitudes → volume. Uses Python MIDI (`music21`, `mido`, or `pyaudio`) to create sound waves. |
| **Visual Ocean** | Renders whales swimming through an ocean whose surface reacts to the quantum amplitudes. Uses `Three.js` (JavaScript) or `OpenGL` (Python/C++). |
| **Entanglement Mode** | When two qubits are entangled, their whale counterparts glow and sing in harmonic resonance — demonstrating nonlocal correlation through sound. |

---

## Tech Stack  
- **Quantum Computing:** [Qiskit](https://qiskit.org) or [Cirq](https://quantumai.google/cirq)  
- **Audio Generation:** `music21`, `mido`, or `pyaudio`  
- **Visualization:** `Three.js` (Web) or `OpenGL` (Python)  
- **Optional ML Add-on:** `TensorFlow Quantum` for adaptive “quantum remixes”  
- **Data (Optional):** Whale song spectrograms from NOAA Open Data  

---

## Features  
- 🎧 Real-time conversion of quantum amplitudes into sound  
- 🐋 Entangled whales that sing together when measured  
- 🌌 Animated ocean reflecting quantum probability waves  
- 🔀 Random-walk “quantum remix” generator for endless melodies  
- 📈 Modular design for music, education, or visualization demos  

---

  ### Project Structure

```
quantum-whalesong/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ Makefile                       # simple build/run commands
├─ quantum_whalesong.py           # main runnable demo (quantum → sound)
│
├─ web/                           # whale & ocean visuals (HTML/JS/CSS)
│  ├─ index.html
│  ├─ styles.css
│  └─ script.js
│
├─ examples/                      # example shell scripts or presets
│  └─ run_bell.sh
│
├─ notebooks/                     # educational or interactive demos
│  └─ 01_qubits_to_music.ipynb
│
├─ src/quantum_whalesong/         # modular Python package (future)
│  ├─ __init__.py
│  ├─ circuits.py
│  ├─ mapping.py
│  ├─ synth.py
│  ├─ io.py
│  └─ cli.py
│
├─ data/                          # optional input data (e.g., NOAA whale songs)
│  └─ .gitkeep
│
└─ exports/                       # generated WAV/JSON outputs (git-ignored)
   └─ .gitkeep
```