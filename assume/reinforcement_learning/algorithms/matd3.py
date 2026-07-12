# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import torch as th
from torch.nn import functional as F

from assume.common.base import LearningStrategy
from assume.reinforcement_learning.algorithms.base_algorithm import A2CAlgorithm
from assume.reinforcement_learning.learning_utils import (
    polyak_update,
)
from assume.reinforcement_learning.neural_network_architecture import CriticTD3

logger = logging.getLogger(__name__)


class TD3(A2CAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradients (TD3).
    Addressing Function Approximation Error in Actor-Critic Methods.
    TD3 is a direct successor of DDPG and improves it using three major tricks:
    clipped double Q-Learning, delayed policy update and target policy smoothing.

    Open AI Spinning guide: https://spinningup.openai.com/en/latest/algorithms/td3.html
    Original paper: https://arxiv.org/pdf/1802.09477.pdf

    Attributes:
        n_updates: Counter for gradient updates performed.
        grad_clip_norm: Maximum gradient norm for clipping.
        critic_architecture_class: Critic network architecture class (CriticTD3).

    Example:
        >>> td3 = TD3(learning_role)
        >>> td3.update_policy()
    """

    def __init__(self, learning_role):
        """Initialize the TD3 algorithm.

        Sets up the algorithm with gradient counters, clipping parameters,
        and critic architecture.

        Args:
            learning_role: Learning role object managing agents and replay buffer.
                Must have off-policy configuration.
        """
        super().__init__(learning_role)

        self.n_updates = 0
        self.grad_clip_norm = 1.0

        # Define the critic architecture class for TD3
        self.critic_architecture_class = CriticTD3

    def get_action(
        self, strategy: "LearningStrategy", obs: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        """Sample an action using the off-policy strategy.

        During learning mode the agent either performs pure-noise initial
        exploration (first N episodes) or uses its deterministic actor plus
        Gaussian action noise.  During evaluation mode the actor is used
        without any noise.

        This default implementation is shared by TD3 and DDPG.  PPO overrides
        it with its own stochastic Gaussian sampling.
        """
        if strategy.learning_mode and not strategy.evaluation_mode:
            if strategy.collect_initial_experience_mode:
                # Pure Gaussian noise for initial random exploration
                noise = th.normal(
                    mean=0.0,
                    std=strategy.exploration_noise_std,
                    size=(strategy.act_dim,),
                    dtype=strategy.float_type,
                    device=strategy.device,
                )
                return noise, noise

            action = strategy.actor(obs).detach()
            noise = strategy.action_noise.noise(
                device=strategy.device, dtype=strategy.float_type
            )
            action = th.clamp(
                action + noise,
                strategy.actor.min_output,
                strategy.actor.max_output,
            )
            return action, noise

        # Evaluation
        action = strategy.actor(obs).detach()
        noise = th.zeros(
            strategy.act_dim, dtype=strategy.float_type, device=strategy.device
        )
        return action, noise

    def update_policy(self):
        """Update the policy using the Twin Delayed Deep Deterministic Policy Gradients (TD3).

        This method performs the policy update step, which involves updating the actor
        (policy) and critic (Q-function) networks using the TD3 algorithm. It iterates
        over the specified number of gradient steps and performs the following for each
        learning strategy:

        1. Sample a batch of transitions from the replay buffer.
        2. Calculate the next actions with added noise using the actor target network.
        3. Compute the target Q-values based on the next states, rewards, and the target critic network.
        4. Compute the critic loss as the mean squared error between current Q-values and target Q-values.
        5. Optimize the critic network by performing a gradient descent step.
        6. Update the actor network if the specified policy delay is reached.
        7. Apply Polyak averaging to update target networks.
        """
        logger.debug("Updating Policy (TD3)")

        # Stack strategies for easier access
        strategies = list(self.learning_role.rl_strats.values())
        n_rl_agents = len(strategies)

        unit_params = [
            {
                u_id: {
                    "actor_loss": None,
                    "actor_total_grad_norm": None,
                    "actor_max_grad_norm": None,
                    "critic_loss": None,
                    "critic_total_grad_norm": None,
                    "critic_max_grad_norm": None,
                }
                for u_id in self.learning_role.rl_strats.keys()
            }
            for _ in range(self.learning_config.off_policy.gradient_steps)
        ]

        # update noise decay and learning rate
        updated_noise_decay = self.learning_role.calc_noise_from_progress(
            self.learning_role.get_progress_remaining()
        )

        learning_rate = self.learning_role.calc_lr_from_progress(
            self.learning_role.get_progress_remaining()
        )

        # PARAMETER-SHARING
        # component 4: loss aggregation
        # critics remain independent, so one optimizer exists per strategy.
        critic_optimizers = [
            strategy.critics.optimizer for strategy in strategies
        ]
        self.update_learning_rate(
            critic_optimizers,
            learning_rate = learning_rate
        )
        # Actors are owned by groups, so update each unique optimizer once.
        self.update_learning_rate(
            list(self.actor_optimizers_by_group.values()),
            learning_rate = learning_rate
        )
        # Exploration noise remains unit-specific.
        for strategy in strategies:
            strategy.action_noise.update_noise_decay(updated_noise_decay)

        for step in range(self.learning_config.off_policy.gradient_steps):
            self.n_updates += 1

            transitions = self.learning_role.buffer.sample(
                self.learning_config.batch_size
            )
            states, actions, next_states, rewards = (
                transitions.observations,
                transitions.actions,
                transitions.next_observations,
                transitions.rewards,
            )

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    th.randn_like(actions)
                    * self.learning_config.off_policy.target_policy_noise
                )
                noise = noise.clamp(
                    -self.learning_config.off_policy.target_noise_clip,
                    self.learning_config.off_policy.target_noise_clip,
                )

                # Select next actions for all agents
                next_actions = th.stack(
                    [
                        (
                            strategy.actor_target(next_states[:, i, :]) + noise[:, i, :]
                        ).clamp(-1, 1)
                        for i, strategy in enumerate(strategies)
                    ]
                )
                next_actions = next_actions.transpose(0, 1).contiguous()
                next_actions = next_actions.view(-1, n_rl_agents * self.act_dim)

            all_actions = actions.view(self.learning_config.batch_size, -1)

            # Precompute unique observation parts for all agents
            unique_obs_from_others = states[
                :, :, self.obs_dim - self.unique_obs_dim :
            ].reshape(self.learning_config.batch_size, n_rl_agents, -1)
            next_unique_obs_from_others = next_states[
                :, :, self.obs_dim - self.unique_obs_dim :
            ].reshape(self.learning_config.batch_size, n_rl_agents, -1)

            #####################################################################
            # CRITIC UPDATE: Accumulate losses for all agents, then backprop once
            #####################################################################

            # Zero-grad for all critics before accumulation
            for strategy in strategies:
                strategy.critics.optimizer.zero_grad(set_to_none=True)

            total_critic_loss = 0.0

            # Loop over all agents and accumulate critic loss
            for i, strategy in enumerate(strategies):
                actor = strategy.actor
                critic = strategy.critics
                critic_target = strategy.target_critics

                # Efficiently extract unique observations from all other agents
                other_unique_obs = th.cat(
                    (unique_obs_from_others[:, :i], unique_obs_from_others[:, i + 1 :]),
                    dim=1,
                )
                other_next_unique_obs = th.cat(
                    (
                        next_unique_obs_from_others[:, :i],
                        next_unique_obs_from_others[:, i + 1 :],
                    ),
                    dim=1,
                )

                # Construct final state representations
                all_states = th.cat(
                    (
                        states[:, i, :].reshape(self.learning_config.batch_size, -1),
                        other_unique_obs.reshape(self.learning_config.batch_size, -1),
                    ),
                    dim=1,
                )
                all_next_states = th.cat(
                    (
                        next_states[:, i, :].reshape(
                            self.learning_config.batch_size, -1
                        ),
                        other_next_unique_obs.reshape(
                            self.learning_config.batch_size, -1
                        ),
                    ),
                    dim=1,
                )

                # Compute the next Q-values: min over all critics targets
                with th.no_grad():
                    next_q_values = th.cat(
                        critic_target(all_next_states, next_actions), dim=1
                    )
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    target_Q_values = (
                        rewards[:, i].unsqueeze(1)
                        + self.learning_config.gamma * next_q_values
                    )

                # Get current Q-values estimates for each critic network
                current_Q_values = critic(all_states, all_actions)

                # Accumulate critic loss for this agent
                critic_loss = sum(
                    F.mse_loss(current_q, target_Q_values)
                    for current_q in current_Q_values
                )

                # Store the critic loss for this unit ID
                unit_params[step][strategy.unit_id]["critic_loss"] = critic_loss.item()
                total_critic_loss += critic_loss

            # Single backward pass for all agents' critics
            total_critic_loss.backward()

            # Clip the gradients and step each critic optimizer
            for strategy in strategies:
                parameters = list(strategy.critics.parameters())

                # Determine clipping statistics
                max_grad_norm = max(p.grad.norm() for p in parameters)

                # Perform clipping
                total_norm = th.nn.utils.clip_grad_norm_(
                    parameters, max_norm=self.grad_clip_norm
                )
                strategy.critics.optimizer.step()

                # Store clipping statistics
                unit_params[step][strategy.unit_id]["critic_total_grad_norm"] = (
                    total_norm
                )
                unit_params[step][strategy.unit_id]["critic_max_grad_norm"] = (
                    max_grad_norm
                )

            ######################################################################
            # ACTOR UPDATE (DELAYED): Accumulate losses for all agents in one pass
            ######################################################################
            if self.n_updates % self.learning_config.off_policy.policy_delay == 0:
                # clear every unique actor-group optimizer exactly once.
                self.zero_actor_group_gradients()
                # preserve one actor loss per unit because every unit has its own critic.
                actor_losses_by_unit: dict[str, th.Tensor] = {}
                for i, strategy in enumerate(strategies):
                    actor = strategy.actor
                    critic = strategy.critics
                    state_i = states[:, i, :]
                    action_i = actor(state_i)
                    other_unique_obs = th.cat(
                        (
                            unique_obs_from_others[:, :i],
                            unique_obs_from_others[:, i+1:]
                        ),
                        dim = 1
                    )
                    all_states_i = th.cat(
                        (
                            state_i.reshape(
                                self.learning_config.batch_size,
                                -1
                            ),
                            other_unique_obs.reshape(
                                self.learning_config.batch_size,
                                -1
                            )
                        ),
                        dim=1
                    )
                    # other agent's replay actions remain fixed.
                    all_actions_clone = actions.clone().detach()
                    # only this unit's action is replaced with its current actor output.
                    all_actions_clone[:, i, :] = action_i
                    all_actions_clone = all_actions_clone.view(
                        self.learning_config.batch_size,
                        -1,
                    )
                    # MATD3 actor uses the first critic output.
                    actor_loss = -critic.q1_forward(
                        all_states_i,
                        all_actions_clone,
                    ).mean()
                    actor_losses_by_unit[strategy.unit_id] = actor_loss
                    # Preserving individual loss logging even when the actor is shared.
                    unit_params[step][strategy.unit_id]["actor_loss"] = actor_loss.item()
                # Convert per-unit losses into one loss per actor group.
                actor_losses_by_group = self.aggregate_actor_losses(
                    actor_losses_by_unit
                )
                # Different groups own different actors, so group losses are summed.
                total_group_actor_loss = th.stack(
                    tuple(actor_losses_by_group.values())
                ).sum()
                # One backward pass accumulates every member's contribution into its corresponding shared actor.
                total_group_actor_loss.backward()
                # This helper clips and steps each unique actor optimizer once.
                actor_gradient_metrics = self.clip_and_step_actor_groups(
                    max_norm=self.grad_clip_norm,
                )
                # Storing metrics per unit. All members of one actor group receive the same group-level gradient statistics.
                for group_id, member_ids in (
                    self.actor_group_registry.groups.items()
                ):
                    group_metrics = actor_gradient_metrics[group_id]
                    for unit_id in member_ids:
                        unit_params[step][unit_id][
                            "actor_total_grad_norm"
                        ] = group_metrics["actor_total_grad_norm"]
                        unit_params[step][unit_id][
                            "actor_max_grad_norm"
                        ] = group_metrics["actor_max_grad_norm"]
                # Critics remain independent, so their source and target parameters are still collected per strategy.
                all_critic_params = []
                all_target_critic_params = []
                for strategy in strategies:
                    all_critic_params.extend(
                        strategy.critics.parameters()
                    )
                    all_target_critic_params.extend(
                        strategy.target_critics.parameters()
                    )
                polyak_update(
                    all_critic_params,
                    all_target_critic_params,
                    self.learning_config.off_policy.tau,
                )
                # Actors and target actors are updated once per group.
                self.soft_update_actor_targets(
                    tau=self.learning_config.off_policy.tau,
                )

        self.learning_role.write_rl_grad_params_to_output(learning_rate, unit_params)
