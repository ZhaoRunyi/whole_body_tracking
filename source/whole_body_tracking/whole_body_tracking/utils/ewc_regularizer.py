from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _select_ewc_parameter_names(policy: nn.Module, actor_only: bool) -> list[str]:
    selected_names: list[str] = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if not actor_only:
            selected_names.append(name)
            continue

        if "critic" in name.lower():
            continue
        selected_names.append(name)
    return selected_names


@dataclass
class EwcConfig:
    enable: bool = False
    lambda_: float = 1.0
    fisher_batches: int = 5
    actor_only: bool = True


class PolicyEwcRegularizer:
    def __init__(self, cfg: EwcConfig):
        self.cfg = cfg
        self.reference_ready = False
        self.fisher_ready = False
        self.last_penalty = 0.0
        self._parameter_names: list[str] = []
        self._theta_star: dict[str, torch.Tensor] = {}
        self._fisher: dict[str, torch.Tensor] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enable)

    def capture_reference(self, policy: nn.Module) -> None:
        if not self.enabled:
            return

        self._parameter_names = _select_ewc_parameter_names(policy, actor_only=self.cfg.actor_only)
        self._theta_star = {}
        self._fisher = {}
        for name, parameter in policy.named_parameters():
            if name not in self._parameter_names:
                continue
            self._theta_star[name] = parameter.detach().clone()
            self._fisher[name] = torch.zeros_like(parameter, device=parameter.device)

        self.reference_ready = True
        self.fisher_ready = False
        self.last_penalty = 0.0

    def estimate_fisher_from_rollout(self, alg, replay_mask_rollout: torch.Tensor | None) -> None:
        if not self.enabled or not self.reference_ready or self.fisher_ready:
            return

        if replay_mask_rollout is None or replay_mask_rollout.numel() == 0:
            return

        policy = getattr(alg, "policy", None)
        storage = getattr(alg, "storage", None)
        optimizer = getattr(alg, "optimizer", None)
        if policy is None or storage is None or optimizer is None:
            raise AttributeError("EWC requires alg.policy, alg.storage, and alg.optimizer.")
        if not hasattr(policy, "act") or not hasattr(policy, "get_actions_log_prob"):
            raise AttributeError("EWC requires policy.act(...) and policy.get_actions_log_prob(...).")

        observations = getattr(storage, "observations", None)
        actions = getattr(storage, "actions", None)
        if observations is None or actions is None:
            raise AttributeError("EWC requires rollout storage with observations and actions tensors.")

        replay_mask = replay_mask_rollout.to(device=observations.device, dtype=torch.float32).view(-1)
        replay_indices = torch.nonzero(replay_mask > 0.5, as_tuple=False).flatten()
        if replay_indices.numel() == 0:
            return

        flat_observations = observations.flatten(0, 1)
        flat_actions = actions.flatten(0, 1)
        replay_observations = flat_observations[replay_indices]
        replay_actions = flat_actions[replay_indices]

        for name in self._fisher:
            self._fisher[name].zero_()

        num_batches = max(int(self.cfg.fisher_batches), 1)
        batch_size = max((replay_observations.shape[0] + num_batches - 1) // num_batches, 1)
        generator = torch.Generator(device=observations.device)
        generator.manual_seed(0)
        shuffled_indices = torch.randperm(replay_observations.shape[0], device=observations.device, generator=generator)

        used_batches = 0
        policy.train()
        for start in range(0, replay_observations.shape[0], batch_size):
            if used_batches >= num_batches:
                break

            batch_indices = shuffled_indices[start : start + batch_size]
            if batch_indices.numel() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            obs_batch = replay_observations[batch_indices]
            action_batch = replay_actions[batch_indices]
            policy.act(obs_batch)
            log_prob = policy.get_actions_log_prob(action_batch)
            fisher_loss = -log_prob.mean()
            fisher_loss.backward()

            for name, parameter in policy.named_parameters():
                if name not in self._fisher or parameter.grad is None:
                    continue
                self._fisher[name] += parameter.grad.detach().pow(2)

            used_batches += 1

        if used_batches <= 0:
            return

        for name in self._fisher:
            self._fisher[name] /= float(used_batches)
        self.fisher_ready = True

    def compute_penalty(self, policy: nn.Module) -> torch.Tensor:
        if not self.enabled or not self.reference_ready or not self.fisher_ready:
            device = next(policy.parameters()).device
            return torch.zeros((), dtype=torch.float32, device=device)

        penalty = None
        for name, parameter in policy.named_parameters():
            if name not in self._fisher:
                continue
            term = (self._fisher[name] * (parameter - self._theta_star[name]).pow(2)).sum()
            penalty = term if penalty is None else penalty + term

        if penalty is None:
            device = next(policy.parameters()).device
            return torch.zeros((), dtype=torch.float32, device=device)
        return penalty

    def apply_penalty_step(self, alg) -> float:
        if not self.enabled or not self.reference_ready or not self.fisher_ready:
            self.last_penalty = 0.0
            return self.last_penalty

        policy = getattr(alg, "policy", None)
        optimizer = getattr(alg, "optimizer", None)
        if policy is None or optimizer is None:
            raise AttributeError("EWC requires alg.policy and alg.optimizer.")

        penalty = self.compute_penalty(policy)
        if float(penalty.detach().item()) <= 0.0:
            self.last_penalty = 0.0
            return self.last_penalty

        optimizer.zero_grad(set_to_none=True)
        loss = float(self.cfg.lambda_) * penalty
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), float(getattr(alg, "max_grad_norm", 1.0)))
        optimizer.step()

        self.last_penalty = float(penalty.detach().item())
        return self.last_penalty
