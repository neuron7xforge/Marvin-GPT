class IntentionEngine:
    def compute_valence_gradient(self, valences: np.ndarray) -> np.ndarray:
        """Compute affective gradient from neighboring agents."""
        gradient = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            left = valences[i - 1] if i > 0 else valences[-1]
            right = valences[(i + 1) % self.n_agents]
            gradient[i] = (right - left) / 2.0
        return gradient / (np.linalg.norm(gradient) + 1e-10)

    def compute_intention_bias(self, valences: np.ndarray, activations: np.ndarray) -> np.ndarray:
        """Generate intention vector based on affective gradient."""
        self.valence_gradient = self.compute_valence_gradient(valences)
        self.intention_bias = INTENTION_GAIN * self.valence_gradient * np.abs(activations)
        return self.intention_bias
