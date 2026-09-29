"""Configuration helpers for self-conditioned mixed-prefix policy training.

The original OAT policy is trained with pure teacher forcing: every action-token
position in the autoregressive prefix comes from the ground-truth tokenizer
output.  At rollout time the prefix is entirely self-generated, so the model
never sees its own (imperfect) history during training.

``SelfConditionedPrefixConfig`` describes the stochastic mixed-prefix scheme
that closes part of that gap:  after predicting register ``k`` the token that is
appended to the prefix is either the clean GT token (a "GT anchor") or the
policy's own greedy prediction.  The decision is drawn independently per sample
and per register, so no fixed anchor position can be memorised.

Only the *inputs* change.  The cross-entropy target is always the clean GT
token, and inference is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


SCHEDULE_CONSTANT = "constant"
SCHEDULE_LINEAR = "linear"
_VALID_SCHEDULES = (SCHEDULE_CONSTANT, SCHEDULE_LINEAR)

PREDICTION_GREEDY = "greedy"
_VALID_PREDICTION_MODES = (PREDICTION_GREEDY,)


@dataclass
class SelfConditionedPrefixConfig:
    """Options for self-conditioned mixed-prefix training.

    Args:
        enabled: When ``False`` the policy uses the original parallel
            teacher-forcing forward pass, bit-for-bit.
        self_prob: Probability that a prefix slot is filled with the policy's
            own prediction instead of the GT token.  Used when
            ``schedule == "constant"``.
        prediction_mode: How the self-conditioned token is drawn.  Only
            ``"greedy"`` (argmax over the action vocabulary) is implemented.
        schedule: ``"constant"`` or ``"linear"``.  ``"linear"`` ramps the
            probability from ``start_prob`` to ``end_prob`` over
            ``warmup_epochs`` epochs (or ``warmup_steps`` optimizer steps when
            ``warmup_epochs`` is ``None``).
        start_prob: Initial probability for the linear schedule.
        end_prob: Final probability for the linear schedule.
        warmup_epochs: Epochs to ramp over. Takes precedence over
            ``warmup_steps`` when both are set.
        warmup_steps: Global steps to ramp over, used when ``warmup_epochs``
            is ``None``.
    """

    enabled: bool = False
    self_prob: float = 0.25
    prediction_mode: str = PREDICTION_GREEDY
    schedule: str = SCHEDULE_CONSTANT
    start_prob: float = 0.0
    end_prob: float = 0.5
    warmup_epochs: Optional[int] = 100
    warmup_steps: Optional[int] = None

    def __post_init__(self):
        self.enabled = bool(self.enabled)
        self.self_prob = float(self.self_prob)
        self.start_prob = float(self.start_prob)
        self.end_prob = float(self.end_prob)
        self.prediction_mode = str(self.prediction_mode)
        self.schedule = str(self.schedule)
        if self.warmup_epochs is not None:
            self.warmup_epochs = int(self.warmup_epochs)
        if self.warmup_steps is not None:
            self.warmup_steps = int(self.warmup_steps)

        if self.schedule not in _VALID_SCHEDULES:
            raise ValueError(
                f"Unknown self_conditioned_prefix.schedule={self.schedule!r}, "
                f"expected one of {_VALID_SCHEDULES}."
            )
        if self.prediction_mode not in _VALID_PREDICTION_MODES:
            raise ValueError(
                "Unknown self_conditioned_prefix.prediction_mode="
                f"{self.prediction_mode!r}, expected one of "
                f"{_VALID_PREDICTION_MODES}."
            )
        for name in ("self_prob", "start_prob", "end_prob"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"self_conditioned_prefix.{name} must be in [0, 1], got {value}."
                )

    @classmethod
    def from_config(cls, config: Optional[Any]) -> "SelfConditionedPrefixConfig":
        """Build from ``None``, a mapping, an ``omegaconf`` node, or an instance."""
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config

        if isinstance(config, Mapping):
            items = dict(config)
        else:
            # omegaconf.DictConfig and friends
            try:
                from omegaconf import DictConfig, OmegaConf
            except ImportError:  # pragma: no cover - omegaconf is a hard dep here
                raise TypeError(
                    f"Cannot interpret self_conditioned_prefix of type {type(config)}."
                )
            if isinstance(config, DictConfig):
                items = OmegaConf.to_container(config, resolve=True)
            else:
                raise TypeError(
                    f"Cannot interpret self_conditioned_prefix of type {type(config)}."
                )

        known = {f for f in cls.__dataclass_fields__}
        unknown = set(items) - known
        if unknown:
            raise ValueError(
                f"Unknown self_conditioned_prefix keys: {sorted(unknown)}. "
                f"Expected a subset of {sorted(known)}."
            )
        return cls(**items)

    def current_self_prob(self, epoch: int = 0, global_step: int = 0) -> float:
        """Resolve the self-prefix probability for the current training progress."""
        if not self.enabled:
            return 0.0
        if self.schedule == SCHEDULE_CONSTANT:
            return self.self_prob

        # linear ramp
        if self.warmup_epochs is not None and self.warmup_epochs > 0:
            frac = float(epoch) / float(self.warmup_epochs)
        elif self.warmup_steps is not None and self.warmup_steps > 0:
            frac = float(global_step) / float(self.warmup_steps)
        else:
            frac = 1.0
        frac = min(max(frac, 0.0), 1.0)
        return self.start_prob + (self.end_prob - self.start_prob) * frac

    def summary(self) -> str:
        if not self.enabled:
            return "self_conditioned_prefix: disabled (original teacher forcing)"
        if self.schedule == SCHEDULE_CONSTANT:
            sched = f"constant p={self.self_prob:.3f}"
        else:
            horizon = (
                f"{self.warmup_epochs} epochs"
                if self.warmup_epochs is not None and self.warmup_epochs > 0
                else f"{self.warmup_steps} steps"
            )
            sched = (
                f"linear p={self.start_prob:.3f} -> {self.end_prob:.3f} over {horizon}"
            )
        return (
            f"self_conditioned_prefix: enabled ({sched}, "
            f"prediction_mode={self.prediction_mode})"
        )
