import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    _HAS_PL = True
except Exception:
    _HAS_PL = False


class TrueQuantumLayer(nn.Module):
    """
    Variational Quantum Circuit (VQC) layer using PennyLane with Qiskit Aer backend.
    Maps a small classical feature vector to expectation values.
    """
    def __init__(self, num_qubits: int = 4, outputs: int = 2):
        super().__init__()
        self.num_qubits = num_qubits
        self.outputs = outputs

        if not _HAS_PL:
            # Fallback: small linear layer if PL is not available
            self.fallback = nn.Linear(num_qubits, outputs)
            return

        # Try Qiskit Aer; fallback to default.qubit
        try:
            self.dev = qml.device("qiskit.aer", wires=num_qubits, backend="aer_simulator", shots=None)
        except Exception:
            self.dev = qml.device("default.qubit", wires=num_qubits, shots=None)

        weight_shapes = {"rot_weights": (num_qubits, 3), "entangling_weights": (num_qubits - 1,)}

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, rot_weights, entangling_weights):
            # Amplitude encoding-lite: angle embedding
            qml.templates.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
            # Variational layers
            for q in range(num_qubits):
                qml.RX(rot_weights[q, 0], wires=q)
                qml.RY(rot_weights[q, 1], wires=q)
                qml.RZ(rot_weights[q, 2], wires=q)
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(entangling_weights[i], wires=i + 1)
            # Two expectation values (pad if more requested)
            measurements = [qml.expval(qml.PauliZ(wires=i)) for i in range(min(outputs, num_qubits))]
            return measurements

        self.circuit = circuit
        self.qlayer = qml.qnn.TorchLayer(self.circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _HAS_PL:
            return self.fallback(x)
        # Expect inputs in range [0, pi]; clamp for safety
        x = torch.clamp(x, 0.0, 3.14159)
        out = self.qlayer(x)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        # If fewer outputs than requested, pad
        if out.shape[-1] < self.outputs:
            pad = self.outputs - out.shape[-1]
            out = F.pad(out, (0, pad), mode="constant", value=0.0)
        return out


class TrueQuantumHybridModel(nn.Module):
    """CNN feature extractor + TrueQuantumLayer classifier."""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, 4)
        self.q = TrueQuantumLayer(num_qubits=4, outputs=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc(x)) * 1.5708  # scale to ~[−pi/2, pi/2]
        x = torch.abs(x)                      # clamp to [0, pi/2] for embedding
        x = self.q(x)
        return x


__all__ = ["TrueQuantumLayer", "TrueQuantumHybridModel"]


