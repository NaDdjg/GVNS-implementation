class ParamController:
    """
    Controls GVNS parameters via discrete actions.

    Safe, bounded, interpretable.
    """

    def __init__(self):

        # BASELINE (reference values)
        self.base_params = {
            "shaking_scale": 1.0,
            "acceptance_delta": 0.05,
            "penalty_scale": 1.0,
            "stagnation_limit": 20,
        }

        # CURRENT VALUES
        self.shaking_scale = self.base_params["shaking_scale"]
        self.acceptance_delta = self.base_params["acceptance_delta"]
        self.penalty_scale = self.base_params["penalty_scale"]
        self.stagnation_limit = self.base_params["stagnation_limit"]

        # STABILITY
        self.inertia = 0.7  # smoothing factor (0=no smoothing, 1=full inertia)

        # BOUNDS 
        self.bounds = {
            "shaking_scale": (0.5, 2.0),
            "acceptance_delta": (0.0, 0.2),
            "penalty_scale": (0.5, 3.0),
            "stagnation_limit": (5, 50),
        }

        self.n_actions = 8
        self.action_names = {
            0: "neutral",
            1: "diversify++",
            2: "diversify+",
            3: "intensify+",
            4: "intensify++",
            5: "feasibility++",
            6: "feasibility+",
            7: "reset",
        }

    # =========================================================
    # CLIP HELPERS
    # =========================================================
    def _clip(self, name, value):
        low, high = self.bounds[name]
        return max(low, min(high, value))
    
    def _smooth_update(self, current, target):
        """Apply inertia smoothing."""
        return self.inertia * current + (1 - self.inertia) * target

    # =========================================================
    # APPLY ACTION
    # =========================================================
    def apply(self, action: int):
        """
        Apply discrete action with smoothing and safety.
        """

        # 0. Neutral (stabilize)
        if action == 0:
            self.shaking_scale = self._smooth_update(
                self.shaking_scale, self.base_params["shaking_scale"]
            )
            self.acceptance_delta = self._smooth_update(
                self.acceptance_delta, self.base_params["acceptance_delta"]
            )
            self.penalty_scale = self._smooth_update(
                self.penalty_scale, self.base_params["penalty_scale"]
            )

        # 1. Diversify++
        elif action == 1:
            self.shaking_scale = self._smooth_update(
                self.shaking_scale, self.shaking_scale * 1.3
            )
            self.acceptance_delta = self._smooth_update(
                self.acceptance_delta, self.acceptance_delta + 0.03
            )

        # 2. Diversify+
        elif action == 2:
            self.shaking_scale = self._smooth_update(
                self.shaking_scale, self.shaking_scale * 1.15
            )

        # 3. Intensify+
        elif action == 3:
            self.shaking_scale = self._smooth_update(
                self.shaking_scale, self.shaking_scale * 0.85
            )
            self.acceptance_delta = self._smooth_update(
                self.acceptance_delta, self.acceptance_delta * 0.7
            )

        # 4. Intensify++
        elif action == 4:
            self.shaking_scale = self._smooth_update(
                self.shaking_scale, self.shaking_scale * 0.7
            )
            self.acceptance_delta = self._smooth_update(
                self.acceptance_delta, self.acceptance_delta * 0.4
            )

        # 5. Feasibility++
        elif action == 5:
            self.penalty_scale = self._smooth_update(
                self.penalty_scale, self.penalty_scale * 1.5
            )

        # 6. Feasibility+
        elif action == 6:
            self.penalty_scale = self._smooth_update(
                self.penalty_scale, self.penalty_scale * 1.2
            )

        # 7. Stagnation reset
        elif action == 7:
            self.shaking_scale = self._smooth_update(
                self.shaking_scale, self.shaking_scale * 1.2
            )
            self.acceptance_delta = self._smooth_update(
                self.acceptance_delta, self.acceptance_delta + 0.02
            )
            self.stagnation_limit = int(self.stagnation_limit * 0.8 + 2)

        else:
            raise ValueError(f"Unknown action {action}")

        # CLIP EVERYTHING
        self._apply_bounds()

    # =========================================================
    # APPLY BOUNDS
    # =========================================================
    def _apply_bounds(self):
        self.shaking_scale = self._clip("shaking_scale", self.shaking_scale)
        self.acceptance_delta = self._clip("acceptance_delta", self.acceptance_delta)
        self.penalty_scale = self._clip("penalty_scale", self.penalty_scale)
        self.stagnation_limit = int(
            self._clip("stagnation_limit", self.stagnation_limit)
        )

    # =========================================================
    # EXPORT (for GVNS use)
    # =========================================================
    def get_params(self):
        """
        Return current parameter values.
        """
        return {
            "shaking_scale": self.shaking_scale,
            "acceptance_delta": self.acceptance_delta,
            "penalty_scale": self.penalty_scale,
            "stagnation_limit": self.stagnation_limit,
        }
    
    # =========================================================
    # DEBUG / LOGGING
    # =========================================================
    def get_state(self):
        """
        Return internal controller state (for logging/debugging).
        """
        return {
            "shaking_scale": self.shaking_scale,
            "acceptance_delta": self.acceptance_delta,
            "penalty_scale": self.penalty_scale,
            "stagnation_limit": self.stagnation_limit,
        }

    # =========================================================
    # RESET
    # =========================================================
    def reset(self):
        self.shaking_scale = self.base_params["shaking_scale"]
        self.acceptance_delta = self.base_params["acceptance_delta"]
        self.penalty_scale = self.base_params["penalty_scale"]
        self.stagnation_limit = self.base_params["stagnation_limit"]