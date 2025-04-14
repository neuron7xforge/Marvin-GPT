## 🧠 IntentionEngine: Affective-Gradient-Based Cognitive Direction Module

### 🔬 Scientific Rationale

The `IntentionEngine` module computes a local cognitive intention vector as a function of:

- the affective valence gradient \((\nabla v)\),
- the activation magnitude \((|\vec{a}|)\),
- a scalar modulation factor \((\gamma)\).

This mechanism models endogenous motivation in self-regulating cognitive systems, independent of external reinforcement.

---

### 📐 Formalized Model

1. **Valence Gradient:**
\[
\nabla v_i = \frac{v_{i+1} - v_{i-1}}{2}
\]

2. **Intention Bias:**
\[
I_i = \gamma \cdot \nabla v_i \cdot |a_i|
\]

Where:
- \(v_i\) = valence of agent \(i\)
- \(a_i\) = activation of agent \(i\)
- \(\gamma = \texttt{INTENTION\_GAIN}\) = intention scaling factor

---

### ✅ Functional Role

The resulting intention vector \( \vec{I} \) modulates internal system dynamics. It facilitates phase reorganization, enhances volitional alignment, and supports goal-directed behavior without external commands.

---

### 🧬 Prototype Code (Python)

```python
import numpy as np

INTENTION_GAIN = 0.2  # Scaling factor for intention bias

class IntentionEngine:
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.valence_gradient = np.zeros(n_agents)
        self.intention_bias = np.zeros(n_agents)

    def compute_valence_gradient(self, valences: np.ndarray) -> np.ndarray:
        """Compute affective gradient using periodic boundary conditions."""
        gradient = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            left = valences[i - 1] if i > 0 else valences[-1]
            right = valences[(i + 1) % self.n_agents]
            gradient[i] = (right - left) / 2.0
        norm = np.linalg.norm(gradient) + 1e-10
        return gradient / norm

    def compute_intention_bias(self, valences: np.ndarray, activations: np.ndarray) -> np.ndarray:
        """Generate intention vector based on valence gradient and activation."""
        self.valence_gradient = self.compute_valence_gradient(valences)
        self.intention_bias = INTENTION_GAIN * self.valence_gradient * np.abs(activations)
        return self.intention_bias
