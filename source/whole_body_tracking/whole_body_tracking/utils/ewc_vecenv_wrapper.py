from __future__ import annotations

import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class EwcReplayMaskRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._ewc_replay_mask_rollout: list[torch.Tensor] = []

    def _current_er_replay_mask(self) -> torch.Tensor:
        replay_mask = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        command_manager = getattr(self.unwrapped, "command_manager", None)
        if command_manager is None:
            return replay_mask

        try:
            motion_command = command_manager.get_term("motion")
        except Exception:
            return replay_mask

        er_env_is_replay = getattr(motion_command, "er_env_is_replay", None)
        if er_env_is_replay is None:
            return replay_mask

        replay_mask[:, 0] = er_env_is_replay.to(device=self.device, dtype=torch.float32)
        return replay_mask

    def reset(self, *args, **kwargs):
        self._ewc_replay_mask_rollout.clear()
        return super().reset(*args, **kwargs)

    def step(self, actions):
        replay_mask = self._current_er_replay_mask()
        result = super().step(actions)
        self._ewc_replay_mask_rollout.append(replay_mask.clone())

        if isinstance(result, tuple) and len(result) in (4, 5):
            extras = result[-1]
            if isinstance(extras, dict):
                extras["er_replay_mask"] = replay_mask[:, 0].clone()
        return result

    def consume_ewc_replay_mask_rollout(self) -> torch.Tensor | None:
        if len(self._ewc_replay_mask_rollout) == 0:
            return None
        replay_mask_rollout = torch.stack(self._ewc_replay_mask_rollout, dim=0)
        self._ewc_replay_mask_rollout.clear()
        return replay_mask_rollout
