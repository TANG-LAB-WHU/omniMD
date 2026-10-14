"""Regenerate original deterministic synthetic weights (not a trained model)."""
import json
from pathlib import Path


def layer(ni, no, phase, activation, residual="none", timestep=None):
    return {
        "weights": [[((i * 7 + j * 3 + phase) % 17 - 8) / 19.0
                     for j in range(no)] for i in range(ni)],
        "bias": [((j * 5 + phase) % 11 - 5) / 23.0 for j in range(no)],
        "activation": activation, "residual": residual, "timestep": timestep,
    }


def embedding(phase):
    return {"layers": [layer(1, 2, phase, "tanh", "concat2", [0.7, 0.9]),
                       layer(2, 4, phase + 1, "tanh", "concat2"),
                       layer(4, 4, phase + 2, "tanh", "identity", [0.2, 0.3, 0.4, 0.5])]}


def fitting(phase):
    return {"layers": [layer(8, 5, phase, "tanh"),
                       layer(5, 5, phase + 1, "tanh", "identity", [0.6] * 5),
                       layer(5, 1, phase + 2, "linear")]}


model = {"format": "omnimd-native-se-e2-a", "version": 1,
         "precision": "float64", "units": "EvAngstrom", "type_map": ["O", "H"],
         "descriptor": {"kind": "se_e2_a", "rcut": 3.0, "rcut_smth": 2.1,
                        "sel": [32, 32], "axis_neuron": 2,
                        "normalization": [
                            {"mean": [0.13, 0, 0, 0], "std": [0.7, 0.9, 0.9, 0.9]},
                            {"mean": [-0.08, 0, 0, 0], "std": [1.1, 0.8, 0.8, 0.8]}],
                        "embedding": [[embedding(2 + i * 5 + j * 3) for j in range(2)]
                                      for i in range(2)]},
         "fitting": [fitting(4), fitting(9)], "atom_energy_bias": [-0.7, 0.2]}
frame = {"positions": [[0.2, 0.3, 0.4], [1.1, 0.5, 0.7], [0.6, 1.4, 0.9], [1.7, 1.3, 1.1]],
         "species": ["O", "H", "H", "O"],
         "cell": [[3.7, 0.0, 0.0], [1.2, 3.4, 0.0], [0.6, 0.5, 3.8]],
         "units": "EvAngstrom"}
root = Path(__file__).resolve().parents[1] / "fixtures"
(root / "synthetic-v1.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
(root / "frame.jsonl").write_text(json.dumps(frame) + "\n", encoding="utf-8")
