from typing import Optional, Tuple
from stable_baselines3 import SAC
import numpy as np


class DeterministicSAC(SAC):
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.policy.predict(observation, state, mask, deterministic=True)
