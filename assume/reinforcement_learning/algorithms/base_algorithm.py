# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import logging
import os

import torch as th
from torch.optim import AdamW

from assume.common.base import LearningStrategy
from assume.reinforcement_learning.algorithms import actor_architecture_aliases
from assume.reinforcement_learning.learning_utils import (
    polyak_update,
    transfer_weights,
)
from assume.reinforcement_learning.parameter_sharing import (
    NetworkGroupRegistry,
    build_group_registry,
)

logger = logging.getLogger(__name__)


class RLAlgorithm:
    """Base reinforcement learning algorithm class.

    This is the foundation class for all Reinforcement Learning algorithms in the framework.
    To implement a custom RL algorithm, subclass this class and override the `update_policy` and `get_action` methods.

    The class provides common functionality for:
    - Learning rate scheduling
    - Parameter saving/loading
    - Device management

    Attributes:
        learning_role: The learning role object containing configuration and strategies.
        learning_config: Configuration parameters from the learning role.
        device: The computation device (CPU/GPU) for tensors.
        float_type: The floating point precision type for computations.
        actor_architecture_class: The actor network architecture class.

    Example:
        >>> class CustomAlgorithm(RLAlgorithm):
        ...     def update_policy(self):
        ...         # Custom policy update logic
        ...         pass
        ...     def get_action(self, strategy, obs):
        ...         # Custom action selection logic
        ...         pass
    """

    def __init__(self, learning_role):
        """Initialize the RL algorithm.

        Args:
            learning_role: Learning role object containing configuration and strategies.
                Must be an instance of the Learning class.
        """
        super().__init__()

        self.learning_role = learning_role
        self.learning_config = learning_role.learning_config

        if self.learning_config.actor_architecture in actor_architecture_aliases.keys():
            self.actor_architecture_class = actor_architecture_aliases[
                self.learning_config.actor_architecture
            ]
        else:
            raise ValueError(
                f"Policy '{self.learning_config.actor_architecture}' unknown. Supported architectures are {list(actor_architecture_aliases.keys())}"
            )

        self.device = self.learning_role.device
        self.float_type = self.learning_role.float_type

    def update_learning_rate(
        self,
        optimizers: list[th.optim.Optimizer] | th.optim.Optimizer,
        learning_rate: float,
    ) -> None:
        """Update optimizer learning rates.

        Sets the learning rate for one or more optimizers. Handles both single
        optimizers and lists of optimizers uniformly.

        Args:
            optimizers: A single optimizer or list of optimizers to update.
            learning_rate: The new learning rate value to set.

        Note:
            Adapted from Stable Baselines 3:
            - https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/base_class.py#L286
            - https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L68

        Example:
            >>> optimizer = AdamW(model.parameters(), lr=0.001)
            >>> algorithm.update_learning_rate(optimizer, 0.0001)
        """

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

    def get_action(
        self, strategy: "LearningStrategy", obs: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        """Sample an action for strategy given observation *obs*.

        Each concrete algorithm overrides this method with its own sampling
        logic.

        Args:
            strategy: The TorchLearningStrategy instance requesting an action.
            obs: Flat observation tensor for a single time-step.

        Returns:
            A (action, noise) tuple, both tensors on the same device as strategy.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement get_action()")

    def update_policy(self) -> None:
        """Update the policy parameters.

        This method must be overridden by subclasses to implement the specific
        policy update logic for each RL algorithm. The base implementation raises
        an error to enforce this requirement.

        Raises:
            NotImplementedError: If called on the base class without override.

        Example:
            >>> class CustomAlgorithm(RLAlgorithm):
            ...     def update_policy(self):
            ...         # Implement algorithm-specific policy update
            ...         pass
        """
        logger.error(
            "No policy update function of the used RL algorithm was defined. "
            "Please define how the policies should be updated in the specific "
            "algorithm you use."
        )

    def load_obj(self, directory: str):
        """Load a serialized object from directory.

        Loads a PyTorch serialized object from the specified directory path.
        The object is loaded onto the device specified by the algorithm's configuration.

        Args:
            directory: Path to the directory containing the serialized object.
                Should point to a valid .pt file.

        Returns:
            object: The deserialized Python object.

        Example:
            >>> model_state = algorithm.load_obj('/path/to/checkpoint.pt')
        """
        return th.load(directory, map_location=self.device, weights_only=True)

    def load_params(self, directory: str) -> None:
        """Load learning parameters from disk.

        Abstract method that should be implemented by subclasses to load
        algorithm-specific parameters from the specified directory.

        Args:
            directory: Path to the directory containing saved parameters.

        Note:
            This is an abstract method that must be overridden by subclasses.
        """

    # PARAMETER-SHARING
    # component 2: grouping
    def build_parameter_sharing_groups(self) -> None:
        """Build actor and critic group mappings for registered strategies."""
        config = self.learning_config.parameter_sharing
        strategies = self.learning_role.rl_strats

        self.actor_group_registry: NetworkGroupRegistry = build_group_registry(
            strategies = strategies,
            mode = config.actor_mode,
            config = config,
            network_name = "actor"
        )

        self.critic_group_registry: NetworkGroupRegistry = build_group_registry(
            strategies = strategies,
            mode = config.critic_mode,
            config = config,
            network_name = "critic"
        )

        # exposing the registries through the Learning Role for diagnostics.
        self.learning_role.actor_group_registry = self.actor_group_registry
        self.learning_role.critic_group_registry = self.critic_group_registry


class A2CAlgorithm(RLAlgorithm):
    """Base actor-critic algorithm class.

    Provides shared functionality for actor-critic reinforcement learning algorithms
    including parameter management, network initialization, and saving/loading utilities.
    This serves as the foundation for algorithms like MATD3, MADDPG, and MAPPO.

    The class handles:
    - Actor and critic network creation and management
    - Target network synchronization (when applicable)
    - Parameter saving and loading
    - Weight transfer between different agent configurations

    Attributes:
        uses_target_networks: Whether this algorithm uses target networks.
            TD3 and DDPG use target networks (True), PPO does not (False).

    Example:
        >>> class ActorCriticAlgorithm(A2CAlgorithm):
        ...     def update_policy(self):
        ...         # Custom actor-critic update logic
        ...         pass
    """

    #: Whether this algorithm uses target networks for stability.
    #: TD3 and DDPG use target networks (True), PPO does not (False).
    uses_target_networks: bool = True
    critic_architecture_class: type[th.nn.Module]

    def __init__(self, learning_role):
        """Initialize the actor-critic algorithm.

        Args:
            learning_role: Learning role object containing configuration and strategies.
        """
        super().__init__(learning_role)

    def save_params(self, directory: str) -> None:
        """Save actor and critic network parameters.

        Saves both actor and critic network parameters to separate subdirectories.
        Creates the directory structure if it doesn't exist.

        Args:
            directory: Base directory path where parameters will be saved.
                Will create 'actors/' and 'critics/' subdirectories.

        Example:
            >>> algorithm.save_params('/path/to/save/directory')
            # Creates:
            # /path/to/save/directory/actors/
            # /path/to/save/directory/critics/
        """
        self.save_critic_params(directory=f"{directory}/critics")
        self.save_actor_params(directory=f"{directory}/actors")

    def save_critic_params(self, directory: str) -> None:
        """Save critic network parameters.

        Saves critic networks, their optimizers, and target critics (if applicable)
        for all registered learning strategies. Also saves agent ID ordering information
        to ensure proper loading.

        Args:
            directory: Directory path where critic parameters will be saved.
                Will be created if it doesn't exist.

        Example:
            >>> algorithm.save_critic_params('/path/to/critics/')
        """
        os.makedirs(directory, exist_ok=True)
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "critic": strategy.critics.state_dict(),
                "critic_optimizer": strategy.critics.optimizer.state_dict(),
            }
            # Only save target critic if this algorithm uses target networks
            if self.uses_target_networks:
                obj["critic_target"] = strategy.target_critics.state_dict()

            path = f"{directory}/critic_{u_id}.pt"
            th.save(obj, path)

        # record the exact order of u_ids and save it with critics to ensure that the same order is used when loading the parameters
        u_id_list = [str(u) for u in self.learning_role.rl_strats.keys()]
        mapping = {"u_id_order": u_id_list}
        map_path = os.path.join(directory, "u_id_order.json")
        with open(map_path, "w") as f:
            json.dump(mapping, f, indent=2)

    # PARAMETER-SHARING
    # component 5: group-aware checkpointing
    def save_actor_params(self, directory: str) -> None:
        """Save actor network parameters.

        Saves actor networks, their optimizers, and target actors (if applicable) for all registered learning strategies. And saves one actor checkpoint per actor-sharing group.

        Args:
            directory: Directory path where actor parameters will be saved.
                Will be created if it doesn't exist.

        Example:
            >>> algorithm.save_actor_params('/path/to/actors/')
        """
        os.makedirs(directory, exist_ok=True)
        manifest = self.build_actor_checkpoint_manifest()
        for group_id, group_data in manifest["groups"].items():
            actor = self.actors_by_group[group_id]
            optimizer = self.actor_optimizers_by_group[group_id]
            checkpoint = {
                "actor": actor.state_dict(),
                "actor_optimizer": optimizer.state_dict()
            }
            if self.uses_target_networks:
                target_actor = self.actor_targets_by_group[group_id]
                checkpoint["actor_target"] = target_actor.state_dict()
            checkpoint_path = os.path.join(
                directory,
                group_data["checkpoint"]
            )
            th.save(
                checkpoint,
                checkpoint_path
            )
        manifest_path = os.path.join(
            directory,
            "sharing_manifest.json"
        )
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(
                manifest,
                manifest_file,
                indent=2,
                sort_keys = True
            )

    def load_params(self, directory: str) -> None:
        """
        Load the parameters of both actor and critic networks.

        This method loads the parameters of both the actor and critic networks associated with the learning role from the specified
        directory. It uses the `load_critic_params` and `load_actor_params` methods to load the respective parameters.

        Args:
            directory: Base directory containing 'actors/' and 'critics/' subdirectories.

        Example:
            >>> algorithm.load_params('/path/to/saved/parameters/')
        """
        self.load_critic_params(directory)
        self.load_actor_params(directory)

    def load_critic_params(self, directory: str) -> None:
        """Load critic network parameters.

        Loads critic networks, target critics (if applicable), and optimizer states
        for each registered agent strategy. Handles cases where the number of agents
        differs between saved and current models by performing intelligent weight transfer.

        Args:
            directory: Base directory containing the 'critics/' subdirectory.

        Note:
            Automatically handles agent count mismatches through weight transfer.
            Preserves the order of agents using saved mapping information.

        Example:
            >>> algorithm.load_critic_params('/path/to/saved/parameters/')
        """
        logger.info("Loading critic parameters...")

        if not os.path.exists(directory):
            logger.warning(
                "Specified directory does not exist. Using randomly initialized critics."
            )
            return

        map_path = os.path.join(directory, "critics", "u_id_order.json")
        if os.path.exists(map_path):
            # read the saved order of u_ids from critics save directory
            with open(map_path) as f:
                loaded_id_order = json.load(f).get("u_id_order", [])
        else:
            logger.warning("No u_id_order.json: assuming same order as current.")
            loaded_id_order = [str(u) for u in self.learning_role.rl_strats.keys()]

        new_id_order = [str(u) for u in self.learning_role.rl_strats.keys()]
        direct_load = loaded_id_order == new_id_order

        if direct_load:
            logger.info("Agents order unchanged. Loading critic weights directly.")
        else:
            logger.info(
                f"Agents length and/or order mismatch: n_old={len(loaded_id_order)}, n_new={len(new_id_order)}. Transferring weights for critics and target critics."
            )

        for u_id, strategy in self.learning_role.rl_strats.items():
            critic_path = os.path.join(directory, "critics", f"critic_{u_id}.pt")
            if not os.path.exists(critic_path):
                logger.warning(f"No saved critic for {u_id}; skipping.")
                continue

            try:
                critic_params = th.load(critic_path, weights_only=True)

                # Required keys depend on whether algorithm uses target networks
                required_keys = ["critic", "critic_optimizer"]
                if self.uses_target_networks:
                    required_keys.append("critic_target")

                for key in required_keys:
                    if key not in critic_params:
                        logger.warning(
                            f"Missing {key} in critic params for {u_id}; skipping."
                        )
                        continue

                if direct_load:
                    strategy.critics.load_state_dict(critic_params["critic"])
                    strategy.critics.optimizer.load_state_dict(
                        critic_params["critic_optimizer"]
                    )
                    # Only load target critic if this algorithm uses target networks
                    if self.uses_target_networks and "critic_target" in critic_params:
                        strategy.target_critics.load_state_dict(
                            critic_params["critic_target"]
                        )
                    logger.debug(f"Loaded critic for {u_id} directly.")
                else:
                    critic_weights = transfer_weights(
                        model=strategy.critics,
                        loaded_state=critic_params["critic"],
                        loaded_id_order=loaded_id_order,
                        new_id_order=new_id_order,
                        obs_base=strategy.obs_dim,
                        act_dim=strategy.act_dim,
                        unique_obs=strategy.unique_obs_dim,
                    )

                    if critic_weights is None:
                        logger.warning(
                            f"Critic weights transfer failed for {u_id}; skipping."
                        )
                        continue

                    strategy.critics.load_state_dict(critic_weights)

                    # Only transfer target critic weights if this algorithm uses target networks
                    if self.uses_target_networks and "critic_target" in critic_params:
                        target_critic_weights = transfer_weights(
                            model=strategy.target_critics,
                            loaded_state=critic_params["critic_target"],
                            loaded_id_order=loaded_id_order,
                            new_id_order=new_id_order,
                            obs_base=strategy.obs_dim,
                            act_dim=strategy.act_dim,
                            unique_obs=strategy.unique_obs_dim,
                        )

                        if target_critic_weights is None:
                            logger.warning(
                                f"Target critic weights transfer failed for {u_id}; skipping."
                            )
                            continue

                        strategy.target_critics.load_state_dict(target_critic_weights)

                    logger.debug(f"Critic weights transferred for {u_id}.")

            except Exception as e:
                logger.warning(f"Failed to load critic for {u_id}: {e}")

    # PARAMETER-SHARING
    # component 5: group-aware checkpointing
    def load_legacy_actor_params(self, directory: str) -> None:
        """Load actor network parameters.

        Loads actor networks, target actors (if applicable), and optimizer states
        for each registered agent strategy from the specified directory.

        Args:
            directory: The directory containing the 'actors/' subdirectory where the parameters should be loaded.

        Example:
            >>> algorithm.load_actor_params('/path/to/saved/parameters/')
        """
        actors_directory = os.path.join(
            directory,
            "actors"
        )
        for group_id, member_ids in self.actor_group_registry.groups.items():
            checkpoint_path = None
            for unit_id in member_ids:
                candidate = os.path.join(
                    actors_directory,
                    f"actor_{unit_id}.pt"
                )
                if os.path.isfile(candidate):
                    checkpoint_path = candidate
                    break
            if checkpoint_path is None:
                logger.warning(f"No legacy actor checkpoint found for groupo '{group_id}'.")
                continue
            actor_params = self.load_obj(directory=checkpoint_path)
            actor = self.actors_by_group[group_id]
            optimizer = self.actor_optimizers_by_group[group_id]
            actor.load_state_dict(actor_params["actor"])
            optimizer.load_state_dict(actor_params["actor_optimizer"])
            if self.uses_target_networks:
                if "actor_target" not in actor_params:
                    raise KeyError(f"Legacy checkpoint for group '{group_id}' does not contain actor_target.")
                self.actor_targets_by_group[group_id].load_state_dict(actor_params["actor_target"])
            actor.loaded = True

    # PARAMETER-SHARING
    # component 5: group-aware checkpointing
    def load_actor_params(self, directory: str) -> None:
        """Load one actor checkpoint per actor-sharing group."""
        logger.info("Loading actor parameters...")
        actors_directory = os.path.join(directory, "actors")
        if not os.path.isdir(actors_directory):
            logger.warning(f"Actor directory '{actors_directory}' does not exist. Using randomly initialized actors.")
            return
        manifest_path = os.path.join(
            actors_directory,
            "sharing_manifest.json"
        )
        # Backward compatibility with per-unit actor checkpoints.
        if not os.path.isfile(manifest_path):
            logger.info("No sharing manifest found. Trying per-unit actor checkpoints.")
            self.load_legacy_actor_params(directory)
            return
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if manifest.get("schema_version") != 1:
            raise ValueError(f"Unsupported actor checkpoint schema version: {manifest.get('schema_version')!r}.")
        saved_algorithm = manifest.get("algorithm")
        if saved_algorithm != self.learning_config.algorithm:
            raise ValueError(
                f"Checkpoint algorithm '{saved_algorithm}' does not match current algorithm '{self.learning_config.algorithm}'."
            )
        saved_mapping = manifest.get("unit_to_actor_group", {})
        current_mapping = dict(self.actor_group_registry.unit_to_group)
        if saved_mapping != current_mapping:
            missing_units = set(current_mapping) - set(saved_mapping)
            unknown_units = set(saved_mapping) - set(current_mapping)
            changed_units = {
                unit_id: {
                    "saved": saved_mapping[unit_id],
                    "current": current_mapping[unit_id]
                }
                for unit_id in (set(saved_mapping) & set(current_mapping)) if saved_mapping[unit_id] != current_mapping[unit_id]
            }
            raise ValueError(f"Actor-sharing mapping does not match checkpoint. Missing units: {sorted(missing_units)}, unknown units: {sorted(unknown_units)}, changed units: {changed_units}.")
        saved_groups = manifest.get("groups", {})
        for group_id in self.actor_group_registry.groups:
            if group_id not in saved_groups:
                raise KeyError(f"Checkpoint manifest does not contain actor group '{group_id}'")
            checkpoint_filename = saved_groups[group_id].get("checkpoint")
            if (
                not isinstance(checkpoint_filename, str)
                or os.path.basename(checkpoint_filename) != checkpoint_filename
            ):
                raise ValueError(f"Invalid checkpoint filename for group '{group_id}': {checkpoint_filename!r}.")
            checkpoint_path = os.path.join(
                actors_directory,
                checkpoint_filename
            )
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Actor checkpoint for group '{group_id}' does not exist: {checkpoint_path}.")
            actor_params = self.load_obj(directory=checkpoint_path)
            actor = self.actors_by_group[group_id]
            optimizer = self.actor_optimizers_by_group[group_id]
            actor.load_state_dict(actor_params["actor"])
            optimizer.load_state_dict(actor_params["actor_optimizer"])
            if self.uses_target_networks:
                if "actor_target" not in actor_params:
                    raise KeyError(f"Actor checkpoint for group '{group_id}' does not contain actor_target.")
                self.actor_targets_by_group[group_id].load_state_dict(actor_params["actor_target"])
            actor.loaded = True

    def initialize_policy(self, actors_and_critics: dict = None) -> None:
        """
        Create actor and critic networks for reinforcement learning.

        If `actors_and_critics` is None, this method creates new actor and critic networks.
        If `actors_and_critics` is provided, it assigns existing networks to the respective attributes.

        Args:
            actors_and_critics: Optional dictionary containing pre-trained networks.
                If None, creates new networks. If provided, assigns existing networks.
                Expected format includes 'actors', 'critics', and optionally
                'actor_targets' and 'target_critics' keys.

        Example:
            >>> # Create new networks
            >>> algorithm.initialize_policy()
            >>>
            >>> # Assign existing networks
            >>> algorithm.initialize_policy(existing_networks_dict)
        """
        # PARAMETER-SHARING
        # component 2: grouping
        self.build_parameter_sharing_groups()

        if actors_and_critics is None:
            self.check_strategy_dimensions()
            self.create_actors()
            self.create_critics()

        else:
            for u_id, strategy in self.learning_role.rl_strats.items():
                strategy.actor = actors_and_critics["actors"][u_id]
                strategy.critics = actors_and_critics["critics"][u_id]

                if self.uses_target_networks:
                    strategy.actor_target = actors_and_critics["actor_targets"][u_id]
                    strategy.target_critics = actors_and_critics["target_critics"][u_id]

            self.obs_dim = actors_and_critics["obs_dim"]
            self.act_dim = actors_and_critics["act_dim"]
            self.unique_obs_dim = actors_and_critics["unique_obs_dim"]
            # PARMETER-SHARING
            # component 3: group architecture
            self.index_existing_group_actors()

    def check_strategy_dimensions(self) -> None:
        """Validate learning strategy dimensions.

        Ensures all registered learning strategies have consistent dimensional
        properties required for centralized critic algorithms. Checks:
        - Observation dimensions
        - Action dimensions
        - Unique observation dimensions
        - Timeseries observation dimensions
        - Foresight parameters
        If not consistent, raises a ValueError. This is important for centralized
        critic algorithms, as it uses a centralized critic that requires consistent
        dimensions across all agents.

        Raises:
            ValueError: If any dimension mismatch is detected across strategies.

        Note:
            This validation is crucial for centralized critic algorithms where
            all agents must have compatible observation and action spaces.
        """
        foresight_list = []
        obs_dim_list = []
        act_dim_list = []
        unique_obs_dim_list = []
        num_timeseries_obs_dim_list = []

        for strategy in self.learning_role.rl_strats.values():
            foresight_list.append(strategy.foresight)
            obs_dim_list.append(strategy.obs_dim)
            act_dim_list.append(strategy.act_dim)
            unique_obs_dim_list.append(strategy.unique_obs_dim)
            num_timeseries_obs_dim_list.append(strategy.num_timeseries_obs_dim)

        if len(set(foresight_list)) > 1:
            raise ValueError(
                f"All foresight values must be the same for all RL agents. The defined learning strategies have the following foresight values: {foresight_list}"
            )
        else:
            self.foresight = foresight_list[0]

        if len(set(act_dim_list)) > 1:
            raise ValueError(
                f"All action dimensions must be the same for all RL agents. The defined learning strategies have the following action dimensions: {act_dim_list}"
            )
        else:
            self.act_dim = act_dim_list[0]

        if len(set(unique_obs_dim_list)) > 1:
            raise ValueError(
                f"All unique_obs_dim values must be the same for all RL agents. The defined learning strategies have the following unique_obs_dim values: {unique_obs_dim_list}"
            )
        else:
            self.unique_obs_dim = unique_obs_dim_list[0]

        if len(set(num_timeseries_obs_dim_list)) > 1:
            raise ValueError(
                f"All num_timeseries_obs_dim values must be the same for all RL agents. The defined learning strategies have the following num_timeseries_obs_dim values: {num_timeseries_obs_dim_list}"
            )
        else:
            self.num_timeseries_obs_dim = num_timeseries_obs_dim_list[0]

        # Check last, as other cases should fail before!
        if len(set(obs_dim_list)) > 1:
            raise ValueError(
                f"All observation dimensions must be the same for all RL agents. The defined learning strategies have the following observation dimensions: {obs_dim_list}"
            )
        else:
            self.obs_dim = obs_dim_list[0]

    def create_actors(self) -> None:
        """Create oen actor, target actor and optimizer per actor group.

        This method initializes actor networks and their corresponding target networks for
        each registered unit strategy. Actors map observations to actions.

        Note:
            All strategies must have the same observation dimension due to the
            centralized critic architecture. Units with different observation
            dimensions require separate learning roles with different critics.

        Example:
            >>> algorithm.create_actors()
            >>> # Creates actor and actor_target for each strategy
        """
        # PARAMETER-SHARING
        # component 3: group architecture
        self.actors_by_group: dict[str, th.nn.Module] = {}
        self.actor_targets_by_group: dict[str, th.nn.Module] = {}
        self.actor_optimizers_by_group: dict[
            str, th.optim.Optimizer
        ] = {}

        for group_id, member_ids in (
            self.actor_group_registry.groups.items()
        ):
            representative_id = member_ids[0]
            representative = self.learning_role.rl_strats[
                representative_id
            ]
            actor = self.create_actor_network(representative)
            optimizer = AdamW(
                actor.parameters(),
                lr = self.learning_role.calc_lr_from_progress(1)
            )
            actor.optimizer = optimizer
            actor.loaded = False
            self.actors_by_group[group_id] = actor
            self.actor_optimizers_by_group[group_id] = optimizer
            actor_target = None
            if self.uses_target_networks:
                actor_target = self.create_actor_network(representative)
                actor_target.load_state_dict(actor.state_dict())
                actor_target.train(mode=False)
                self.actor_targets_by_group[group_id] = actor_target
            
            for unit_id in member_ids:
                strategy = self.learning_role.rl_strats[unit_id]
                strategy.actor = actor
                if self.uses_target_networks:
                    strategy.actor_target = actor_target

    def create_critics(self) -> None:
        """Create critic networks for all learning strategies.

        Initializes critic networks and their corresponding target networks for
        each registered agent strategy. Critics evaluate state-action pairs.

        Note:
            All strategies must have the same observation dimension due to the
            centralized critic architecture. Units with different observation
            dimensions require separate learning roles with different critics.

        Example:
            >>> algorithm.create_critics()
            >>> # Creates critics and target_critics for each strategy
        """
        n_agents = len(self.learning_role.rl_strats)

        for strategy in self.learning_role.rl_strats.values():
            strategy.critics = self.critic_architecture_class(
                n_agents=n_agents,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                unique_obs_dim=self.unique_obs_dim,
                float_type=self.float_type,
            ).to(self.device)

            if self.uses_target_networks:
                strategy.target_critics = self.critic_architecture_class(
                    n_agents=n_agents,
                    obs_dim=self.obs_dim,
                    act_dim=self.act_dim,
                    unique_obs_dim=self.unique_obs_dim,
                    float_type=self.float_type,
                ).to(self.device)

                strategy.target_critics.load_state_dict(strategy.critics.state_dict())
                strategy.target_critics.train(mode=False)

            strategy.critics.optimizer = AdamW(
                strategy.critics.parameters(),
                lr=self.learning_role.calc_lr_from_progress(
                    1
                ),  # 1 = 100% of simulation remaining, uses learning_rate from config as starting point
            )

    def extract_policy(self) -> dict:
        """Extract all policy networks.

        Collects actor and critic networks from all learning strategies into
        a structured dictionary. Includes both primary and target networks.

        Returns:
            Dictionary containing all network components organized by type:
                - 'actors': Primary actor networks
                - 'actor_targets': Target actor networks
                - 'critics': Primary critic networks
                - 'target_critics': Target critic networks
                - Dimension information for reconstruction

        Example:
            >>> policy_dict = algorithm.extract_policy()
            >>> # Contains all networks ready for saving or transfer
        """
        actors = {}
        critics = {}
        if self.uses_target_networks:
            actor_targets = {}
            target_critics = {}

        for u_id, strategy in self.learning_role.rl_strats.items():
            actors[u_id] = strategy.actor
            critics[u_id] = strategy.critics
            if self.uses_target_networks:
                actor_targets[u_id] = strategy.actor_target
                target_critics[u_id] = strategy.target_critics

        actors_and_critics = {
            "actors": actors,
            "critics": critics,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "unique_obs_dim": self.unique_obs_dim,
        }

        if self.uses_target_networks:
            actors_and_critics["actor_targets"] = actor_targets
            actors_and_critics["target_critics"] = target_critics

        return actors_and_critics

    # PARAMETER-SHARING
    # component 3: group architecture
    def create_actor_network(
        self,
        strategy: LearningStrategy,
    ) -> th.nn.Module:
        """Construct one deterministic actor network."""

        return self.actor_architecture_class(
            obs_dim = self.obs_dim,
            act_dim = self.act_dim,
            float_type = self.float_type,
            unique_obs_dim = self.unique_obs_dim,
            num_timeseries_obs_dim = strategy.num_timeseries_obs_dim
        ).to(self.device)

    # PARMETER-SHARING
    # component 3: group architecture
    def index_existing_group_actors(self) -> None:
        self.actors_by_group = {}
        self.actor_targets_by_group = {}
        self.actor_optimizers_by_group = {}

        for group_id, member_ids in self.actor_group_registry.groups.items():
            first_strategy = self.learning_role.rl_strats[member_ids[0]]
            actor = first_strategy.actor
            for unit_id in member_ids[1:]:
                member_actor = self.learning_role.rl_strats[
                    unit_id
                ].actor
                if member_actor is not actor:
                    raise ValueError(
                        f"Actor group '{group_id}' did not preserve shared object identity between episodes."
                    )
            self.actors_by_group[group_id] = actor
            self.actor_optimizers_by_group[group_id] = actor.optimizer
            if self.uses_target_networks:
                target = first_strategy.actor_target
                for unit_id in member_ids[1:]:
                    member_target = self.learning_role.rl_strats[unit_id].actor_target
                    if member_target is not target:
                        raise ValueError(
                            f"Actor target group '{group_id}' did not preserve shared object identity."
                        )
                self.actor_targets_by_group[group_id] = target

    # PARAMETER-SHARING
    # component 4: loss aggregation
    def aggregate_actor_losses(
        self,
        losses_by_unit
    ):
        group_losses = {}
        for group_id, member_ids in self.actor_group_registry.groups.items():
            member_losses = [
                losses_by_unit[unit_id] for unit_id in member_ids
            ]
            stacked_losses = th.stack(member_losses)
            method = self.learning_config.parameter_sharing.loss_aggregation
            if method == "mean":
                group_losses[group_id] = stacked_losses.mean()
            elif method == "sum":
                group_losses[group_id] = stacked_losses.sum()
            else:
                raise NotImplementedError(
                    f"loss_aggregation='{method}' is not implemented yet."
                )
        return group_losses

    # PARAMETER-SHARING
    # component 4: loss aggregation
    def zero_actor_group_gradients(self):
        for optimizer in self.actor_optimizers_by_group.values():
            optimizer.zero_grad(set_to_none=True)

    # PARAMETER-SHARING
    # component 4: loss aggregation
    def clip_and_step_actor_groups(
        self,
        max_norm: float,
    ) -> dict[str, dict[str, float]]:
        """Clip gradients and step every unique actor-group optimizer once."""
        gradient_metrics: dict[str, dict[str, float]] = {}
        for group_id, actor in self.actors_by_group.items():
            # Only parameters that received gradients should participate.
            parameters_with_grad = [
                parameter for parameter in actor.parameters() if parameter.grad is not None
            ]
            if not parameters_with_grad:
                raise RuntimeError(
                    f"Actor group '{group_id}' has no gradients. Make sure its member losses were included before backward()."
                )
            # Largest individual parameter-gradient norm before clipping.
            max_grad_norm = max(
                parameter.grad.detach().norm().item()
                for parameter in parameters_with_grad
            )
            # clip_grad_norm_ returns the total norm measured before clipping.
            total_grad_norm = th.nn.utils.clip_grad_norm_(
                parameters_with_grad,
                max_norm=max_norm,
            )
            optimizer = self.actor_optimizers_by_group[group_id]
            # This is the one and only optimizer step for this group.
            optimizer.step()
            gradient_metrics[group_id] = {
                "actor_total_grad_norm": float(total_grad_norm.item()),
                "actor_max_grad_norm": float(max_grad_norm),
            }
        return gradient_metrics

    # PARAMETER-SHARING
    # component 4: loss aggregation
    def soft_update_actor_targets(
        self,
        tau: float
    ) -> None:
        """Update every unique target actor exactly once. If an algorithm does not use target actors, this method safely returns when target networks are disabled."""
        if not self.uses_target_networks:
            return
        if not 0.0 <= tau <= 1.0:
            raise ValueError(
                f"Polyak coefficient tau must be between 0 and 1, got {tau}."
            )
        actor_group_ids = set(self.actors_by_group)
        target_group_ids = set(self.actor_targets_by_group)
        if actor_group_ids != target_group_ids:
            missing_targets = actor_group_ids - target_group_ids
            unexpected_targets = target_group_ids - actor_group_ids
            raise RuntimeError(
                f"Actor and target-actor group mappings do not match. Missing targets: {sorted(missing_targets)}, Unexpected targets: {sorted(unexpected_targets)}"
            )
        for group_id, actor in self.actors_by_group.items():
            target_actor = self.actor_targets_by_group[group_id]
            polyak_update(
                actor.parameters(),
                target_actor.parameters(),
                tau
            )

    # PARAMETER-SHARING
    # component 5: group-aware checkpointing
    def build_actor_checkpoint_manifest(self) -> dict:
        """describing actor-group ownership and checkpoint filenames."""
        config = self.learning_config.parameter_sharing
        groups = {}
        for index, (group_id, member_ids) in enumerate(
            self.actor_group_registry.groups.items()
        ):
            groups[group_id] = {
                "members": list(member_ids),
                "checkpoint": f"actor_group_{index}.pt"
            }
        return {
            "schema_version": 1,
            "algorithm": self.learning_config.algorithm,
            "parameter_sharing_enabled": config.enabled,
            "actor_mode": config.actor_mode,
            "grouping_method": config.grouping_method,
            "grouping_feature": config.grouping_feature,
            "loss_aggregation": config.loss_aggregation,
            "uses_target_networks": self.uses_target_networks,
            "unit_to_actor_group": dict(
                self.actor_group_registry.unit_to_group
            ),
            "groups": groups
        }
