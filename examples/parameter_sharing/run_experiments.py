# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import json
import shutil
import time
import yaml

from dataclasses import dataclass

from assume import World
from assume.scenario.loader_csv import (
    load_scenario_folder,
    run_learning
)
from assume.strategies.learning_strategies import EnergyLearningStrategy

from dataclasses import replace

from assume.reinforcement_learning.parameter_sharing import (
    build_conditioning_registry,
)

from sqlalchemy import create_engine

EXAMPLES_DIRECTORY = Path(__file__).resolve().parents[1]
INPUTS_PATH = EXAMPLES_DIRECTORY / "inputs"
PARAMETER_SHARING_DIRECTORY = Path(__file__).resolve().parent

EXAMPLE_NAME = "example_02b"
STUDY_CASE = "base"

OUTPUT_DIRECTORY = EXAMPLES_DIRECTORY / "outputs"
DATABASE_DIRECTORY = OUTPUT_DIRECTORY / "local_db"

ALGORITHMS = (
    "mappo",
    "maddpg",
    "matd3",
)

EXAMPLES = (
    # "example_02a",
    "example_02b",
    # "example_02c",
)

TRAINING_EPISODES = 50
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)

REFERENCE_PRICE_DIRECTORY = (
    PARAMETER_SHARING_DIRECTORY / "reference_prices"
)

EXPERIMENT_DIRECTORY = (
    OUTPUT_DIRECTORY / "parameter_sharing_experiments"
)

SHARING_VARIANTS = {
    "independent": {
        "enabled": False,
        "actor_mode": "independent",
        "critic_mode": "independent",
        "grouping_method": "individual",
        "grouping_feature": "technology",
        "conditioning_method": "none",
        "context_architecture": "concatenation",
        "loss_aggregation": "mean",
        "unit_to_group": {},
        "context_features": [],
        "n_clusters": None,
        "max_group_size": None,
    },
    "full_none": {
        "enabled": True,
        "actor_mode": "full",
        "critic_mode": "independent",
        "grouping_method": "all",
        "grouping_feature": "technology",
        "conditioning_method": "none",
        "context_architecture": "concatenation",
        "loss_aggregation": "mean",
        "unit_to_group": {},
        "context_features": [],
        "n_clusters": None,
        "max_group_size": None,
    },
    "full_one_hot": {
        "enabled": True,
        "actor_mode": "full",
        "critic_mode": "independent",
        "grouping_method": "all",
        "grouping_feature": "technology",
        "conditioning_method": "one_hot_unit_id",
        "context_architecture": "concatenation",
        "loss_aggregation": "mean",
        "unit_to_group": {},
        "context_features": [],
        "n_clusters": None,
        "max_group_size": None,
    },
    "full_semantic": {
        "enabled": True,
        "actor_mode": "full",
        "critic_mode": "independent",
        "grouping_method": "all",
        "grouping_feature": "technology",
        "conditioning_method": "semantic_context",
        "context_architecture": "concatenation",
        "loss_aggregation": "mean",
        "unit_to_group": {},
        "context_features": [
            "technology",
            "max_power",
            "min_power",
            "efficiency",
            "emission_factor",
        ],
        "n_clusters": None,
        "max_group_size": None,
    },
    "full_id_context": {
        "enabled": True,
        "actor_mode": "full",
        "critic_mode": "independent",
        "grouping_method": "all",
        "grouping_feature": "technology",
        "conditioning_method": "unit_id_and_context",
        "context_architecture": "concatenation",
        "loss_aggregation": "mean",
        "unit_to_group": {},
        "context_features": [
            "technology",
            "max_power",
            "min_power",
            "efficiency",
            "emission_factor",
        ],
        "n_clusters": None,
        "max_group_size": None,
    },
}

@dataclass(frozen=True)
class ExperimentSpec:
    example_name: str
    algorithm: str
    variant_name: str
    seed: int
    training_episodes: int

    @property
    def run_id(self) -> str:
        return (
            f"{self.example_name}"
            f"__{self.algorithm}"
            f"__{self.variant_name}"
            f"__seed_{self.seed}"
        )

def create_world() -> World:
    """create an empty assume world with a local sqlite database."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DATABASE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    database_path = DATABASE_DIRECTORY / "parameter_sharing_test.db"

    return World(
        database_uri = f"sqlite:///{database_path}",
        export_csv_path = str(OUTPUT_DIRECTORY / "simulation")
    )

def load_example(world: World) -> None:
    """Load example using assume's standard scenario loader."""
    world.bidding_strategies[
        "powerplant_energy_learning"
    ] = EnergyLearningStrategy

    load_scenario_folder(
        world = world,
        inputs_path = str(INPUTS_PATH),
        scenario = EXAMPLE_NAME,
        study_case = STUDY_CASE
    )

def configure_temporary_scenario(
    spec: ExperimentSpec,
    temporary_root: Path,
    policy_directory: Path,
) -> tuple[Path, str]:
    """Copy and configure one scenario before ASSUME loads it."""

    source = INPUTS_PATH / spec.example_name
    destination = temporary_root / spec.example_name

    shutil.copytree(source, destination)

    config_path = destination / "config.yaml"

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    scenario_config = config[STUDY_CASE]
    learning_config = scenario_config["learning_config"]

    scenario_config["seed"] = spec.seed

    learning_config["algorithm"] = spec.algorithm
    learning_config["actor_architecture"] = "mlp"
    learning_config["training_episodes"] = (
        spec.training_episodes
    )
    learning_config["continue_learning"] = False
    learning_config["trained_policies_load_path"] = None
    learning_config["trained_policies_save_path"] = str(
        policy_directory
    )
    learning_config["parameter_sharing"] = deepcopy(
        SHARING_VARIANTS[spec.variant_name]
    )

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    return temporary_root, spec.example_name

def print_parameter_sharing_config(world: World) -> None:
    config = world.learning_role.learning_config.parameter_sharing

    print("\nParameter-sharing configuration")
    print(f"Configuration class: {type(config).__name__}")
    print(f"Enabled: {config.enabled}")
    print(f"Actor mode: {config.actor_mode}")
    print(f"Critic mode: {config.critic_mode}")
    print(f"Grouping method: {config.grouping_method}")
    print(f"Conditioning: {config.conditioning_method}")
    print(f"Context architecture {config.context_architecture}")
    print(f"Loss aggregation: {config.loss_aggregation}")

def inspect_group_registries(world: World) -> None:
    """Build and display logical actor and critic groups."""

    algorithm = world.learning_role.rl_algorithm
    algorithm.build_parameter_sharing_groups()

    actor_registry = algorithm.actor_group_registry
    critic_registry = algorithm.critic_group_registry

    print("\nActor groups")
    for group_id, members in actor_registry.groups.items():
        print(f"{group_id}: {list(members)}")

    print("\nCritic groups")
    for group_id, members in critic_registry.groups.items():
        print(f"{group_id}: {list(members)}")

    expected_units = set(world.learning_role.rl_strats)

    assert set(actor_registry.unit_to_group) == expected_units
    assert set(critic_registry.unit_to_group) == expected_units

    # Current example has one manually defined actor group.
    # assert len(actor_registry.groups) == 1

    # Critics remain independent: one logical critic group per unit.
    # assert len(critic_registry.groups) == len(expected_units)

def load_learning_results() -> pd.DataFrame:
    database_path = DATABASE_DIRECTORY / "parameter_sharing_test.db"
    engine = create_engine(f"sqlite:///{database_path}")

    query = """
    select
        episode,
        unit,
        reward,
        profit,
        actions_0,
        actions_1,
        evaluation_mode
    from rl_params
    where simulation = :simulation_id
    order by episode, datetime, unit
    """

    return pd.read_sql_query(
        query,
        engine,
        params = {"simulation_id": f"{EXAMPLE_NAME}_{STUDY_CASE}"}
    )

def prepare_training_curves(
    results: pd.DataFrame,
    max_bid_price: float,
) -> tuple[pd.Series, pd.DataFrame]:

    training = results[results["evaluation_mode"] == 0].copy()

    if training.empty:
        raise ValueError("No training records found in rl_params")

    # Calculate total episode return for each unit, then average across units.
    unit_episode_returns = (
        training
        .groupby(["episode", "unit"])["reward"]
        .sum()
    )

    average_episode_return = (
        unit_episode_returns
        .groupby("episode")
        .mean()
    )

    # EnergyLearningStrategy converts its two actions into two bid prices.
    training["bid_price_0"] = (
        training["actions_0"] * max_bid_price
    )
    training["bid_price_1"] = (
        training["actions_1"] * max_bid_price
    )

    # The flexible bid uses the larger of the two action-derived prices.
    training["flexible_bid_price"] = training[
        ["bid_price_0", "bid_price_1"]
    ].max(axis=1)

    average_bid_prices = (
        training
        .groupby(["episode", "unit"])["flexible_bid_price"]
        .mean()
        .unstack("unit")
    )

    return average_episode_return, average_bid_prices

def plot_results(
    episode_returns: pd.Series,
    bid_prices: pd.DataFrame,
    marginal_cost: float,
) -> None:

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        constrained_layout=True,
    )

    reward_axis = axes[0]
    bid_axis = axes[1]

    reward_axis.plot(
        episode_returns.index,
        episode_returns.values,
        color="tab:blue",
        linewidth=1.5,
    )
    reward_axis.set_title("Average Training Return per Unit")
    reward_axis.set_xlabel("Training episode")
    reward_axis.set_ylabel("Average episode return")
    reward_axis.grid(alpha=0.3)

    for unit_id in bid_prices.columns:
        bid_axis.plot(
            bid_prices.index,
            bid_prices[unit_id],
            label=unit_id,
            linewidth=1.2,
        )

    bid_axis.axhline(
        marginal_cost,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Marginal cost: {marginal_cost:.2f} EUR/MWh",
    )

    bid_axis.set_title(
        "Average Flexible Bid Price versus Marginal Cost"
    )
    bid_axis.set_xlabel("Training episode")
    bid_axis.set_ylabel("Price [EUR/MWh]")
    bid_axis.grid(alpha=0.3)
    bid_axis.legend(ncol=2)

    output_path = (
        OUTPUT_DIRECTORY / "parameter_sharing_test_results.png"
    )

    figure.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")

    plt.show()

def inspect_actor_ownership(world: World) -> None:

    algorithm = world.learning_role.rl_algorithm
    algorithm.initialize_policy()

    actor_registry = algorithm.actor_group_registry

    print("\nActor ownership")

    for group_id, member_ids in actor_registry.groups.items():
        actor_ids = {
            id(world.learning_role.rl_strats[unit_id].actor)
            for unit_id in member_ids
        }

        optimizer_ids = {
            id(
                world.learning_role
                .rl_strats[unit_id]
                .actor
                .optimizer
            )
            for unit_id in member_ids
        }

        print(
            f"{group_id}: "
            f"members={list(member_ids)}, "
            f"actors={actor_ids}, "
            f"optimizers={optimizer_ids}"
        )

        assert len(actor_ids) == 1
        assert len(optimizer_ids) == 1

        if algorithm.uses_target_networks:
            target_ids = {
                id(
                    world.learning_role
                    .rl_strats[unit_id]
                    .actor_target
                )
                for unit_id in member_ids
            }

            assert len(target_ids) == 1

    critic_ids = {
        id(strategy.critics)
        for strategy in world.learning_role.rl_strats.values()
    }

    assert len(critic_ids) == len(
        world.learning_role.rl_strats
    )

def inspect_actor_loss_aggregation(
    world: World,
) -> None:
    algorithm = world.learning_role.rl_algorithm
    registry = algorithm.actor_group_registry

    losses_by_unit = {
        unit_id: th.tensor(
            float(index + 1),
            dtype=world.learning_role.float_type,
            requires_grad=True,
        )
        for index, unit_id in enumerate(
            world.learning_role.rl_strats
        )
    }

    losses_by_group = algorithm.aggregate_actor_losses(
        losses_by_unit
    )

    reduction = (
        world.learning_role
        .learning_config
        .parameter_sharing
        .loss_aggregation
    )

    for group_id, member_ids in registry.groups.items():
        expected_members = th.stack(
            tuple(
                losses_by_unit[unit_id]
                for unit_id in member_ids
            )
        )

        if reduction == "mean":
            expected = expected_members.mean()
        elif reduction == "sum":
            expected = expected_members.sum()
        else:
            raise NotImplementedError(
                f"Test does not support reduction '{reduction}'."
            )

        assert th.allclose(
            losses_by_group[group_id],
            expected,
        )

    print("\nComponent 4 actor-loss aggregation test passed.")

def inspect_actor_checkpoint_roundtrip(
    world: World,
) -> None:
    algorithm = world.learning_role.rl_algorithm

    original_optimizer_states = {
        group_id: deepcopy(optimizer.state_dict())
        for group_id, optimizer
        in algorithm.actor_optimizers_by_group.items()
    }

    original_actor_states = {
        group_id: {
            name: tensor.detach().clone()
            for name, tensor in actor.state_dict().items()
        }
        for group_id, actor
        in algorithm.actors_by_group.items()
    }

    original_target_states = {}

    if algorithm.uses_target_networks:
        original_target_states = {
            group_id: {
                name: tensor.detach().clone()
                for name, tensor
                in target.state_dict().items()
            }
            for group_id, target
            in algorithm.actor_targets_by_group.items()
        }

    with TemporaryDirectory(
        prefix="assume-parameter-sharing-checkpoint-"
    ) as temporary_directory:
        actors_directory = (
            Path(temporary_directory) / "actors"
        )

        algorithm.save_actor_params(
            directory=str(actors_directory)
        )

        manifest_path = (
            actors_directory / "sharing_manifest.json"
        )

        assert manifest_path.is_file()

        with manifest_path.open(
            encoding="utf-8"
        ) as manifest_file:
            manifest = json.load(manifest_file)

        assert manifest["schema_version"] == 1

        assert manifest["unit_to_actor_group"] == dict(
            algorithm.actor_group_registry.unit_to_group
        )

        checkpoint_files = list(
            actors_directory.glob(
                "actor_group_*.pt"
            )
        )

        assert len(checkpoint_files) == len(
            algorithm.actor_group_registry.groups
        )

        manifest_checkpoint_files = {
            group_data["checkpoint"]
            for group_data in manifest["groups"].values()
        }

        actual_checkpoint_files = {
            path.name
            for path in checkpoint_files
        }

        assert (
            actual_checkpoint_files
            == manifest_checkpoint_files
        )

        with th.no_grad():
            for actor in algorithm.actors_by_group.values():
                for parameter in actor.parameters():
                    parameter.add_(1.0)

            for target in (
                algorithm.actor_targets_by_group.values()
            ):
                for parameter in target.parameters():
                    parameter.add_(1.0)

        for optimizer in (
            algorithm.actor_optimizers_by_group.values()
        ):
            optimizer.param_groups[0]["lr"] *= 10

        algorithm.load_actor_params(
            directory=temporary_directory
        )

        for group_id, actor in (
            algorithm.actors_by_group.items()
        ):
            restored_state = actor.state_dict()

            for name, expected_tensor in (
                original_actor_states[group_id].items()
            ):
                assert th.equal(
                    restored_state[name],
                    expected_tensor,
                )

        for group_id, optimizer in (
            algorithm.actor_optimizers_by_group.items()
        ):
            restored = optimizer.state_dict()
            expected = original_optimizer_states[group_id]

            assert (
                restored["param_groups"]
                == expected["param_groups"]
            )
            assert (
                restored["state"].keys()
                == expected["state"].keys()
            )

            for parameter_id, expected_state in (
                expected["state"].items()
            ):
                restored_state = restored["state"][
                    parameter_id
                ]

                for state_name, expected_value in (
                    expected_state.items()
                ):
                    restored_value = restored_state[
                        state_name
                    ]

                    if th.is_tensor(expected_value):
                        assert th.equal(
                            restored_value,
                            expected_value,
                        )
                    else:
                        assert (
                            restored_value
                            == expected_value
                        )

        if algorithm.uses_target_networks:
            for group_id, target in (
                algorithm.actor_targets_by_group.items()
            ):
                restored_state = target.state_dict()

                for name, expected_tensor in (
                    original_target_states[
                        group_id
                    ].items()
                ):
                    assert th.equal(
                        restored_state[name],
                        expected_tensor,
                    )

        for member_ids in (
            algorithm.actor_group_registry.groups.values()
        ):
            actor_ids = {
                id(
                    world.learning_role
                    .rl_strats[unit_id]
                    .actor
                )
                for unit_id in member_ids
            }

            assert len(actor_ids) == 1

    print(
        "\nComponent 5 actor-checkpoint round trip passed."
    )

def inspect_conditioning_registry(
    world: World,
) -> None:

    algorithm = world.learning_role.rl_algorithm
    strategies = world.learning_role.rl_strats
    base_config = (
        world.learning_role
        .learning_config
        .parameter_sharing
    )
    unit_ids = tuple(strategies)

    algorithm.build_parameter_sharing_conditioning()

    assert (
        algorithm.conditioning_registry
        is world.learning_role.conditioning_registry
    )

    live_registry = algorithm.conditioning_registry

    for unit_id in unit_ids:
        unbatched = algorithm.conditioning_tensor(unit_id)
        batched = algorithm.conditioning_tensor(
            unit_id,
            batch_size=3,
        )

        assert unbatched.shape == (
            live_registry.context_dim,
        )
        assert batched.shape == (
            3,
            live_registry.context_dim,
        )
        assert unbatched.device == algorithm.device
        assert batched.device == algorithm.device
        assert unbatched.dtype == algorithm.float_type
        assert batched.dtype == algorithm.float_type

    try:
        algorithm.conditioning_tensor(
            unit_ids[0],
            batch_size=0,
        )
    except ValueError as error:
        assert "batch_size" in str(error)
    else:
        raise AssertionError(
            "Invalid conditioning batch size was not rejected"
        )

    # No conditioning preserves the current actor input.
    none_config = replace(
        base_config,
        conditioning_method="none",
    )

    none_registry = build_conditioning_registry(
        strategies,
        none_config,
    )

    assert none_registry.context_dim == 0

    for unit_id in unit_ids:
        assert none_registry.vector_for(unit_id) == ()

    identity_config = replace(
        base_config,
        conditioning_method="one_hot_unit_id",
    )

    identity_registry = build_conditioning_registry(
        strategies,
        identity_config,
    )

    assert identity_registry.context_dim == len(unit_ids)

    assert len(
        {
            identity_registry.vector_for(unit_id)
            for unit_id in unit_ids
        }
    ) == len(unit_ids)

    for unit_id in unit_ids:
        vector = identity_registry.vector_for(unit_id)
        assert sum(vector) == 1.0

    semantic_config = replace(
        base_config,
        conditioning_method="semantic_context",
        context_features=[
            "technology",
            "max_power",
            "min_power",
            "efficiency",
            "emission_factor",
        ],
    )

    semantic_registry = build_conditioning_registry(
        strategies,
        semantic_config,
    )

    assert semantic_registry.context_dim > 0

    for unit_id in unit_ids:
        vector = semantic_registry.vector_for(unit_id)

        assert len(vector) == semantic_registry.context_dim
        assert all(np.isfinite(value) for value in vector)
        assert all(0.0 <= value <= 1.0 for value in vector)

    combined_config = replace(
        semantic_config,
        conditioning_method="unit_id_and_context",
    )

    combined_registry = build_conditioning_registry(
        strategies,
        combined_config,
    )

    assert combined_registry.context_dim == (
        identity_registry.context_dim
        + semantic_registry.context_dim
    )

    for unit_id in unit_ids:
        assert combined_registry.vector_for(unit_id) == (
            identity_registry.vector_for(unit_id)
            + semantic_registry.vector_for(unit_id)
        )

    # Rebuilding must be deterministic.
    repeated_registry = build_conditioning_registry(
        strategies,
        combined_config,
    )

    assert (
        repeated_registry.unit_to_vector
        == combined_registry.unit_to_vector
    )
    assert (
        repeated_registry.feature_names
        == combined_registry.feature_names
    )

    # A missing metadata feature must fail clearly.
    missing_feature_config = replace(
        base_config,
        conditioning_method="semantic_context",
        context_features=["does_not_exist"],
    )

    try:
        build_conditioning_registry(
            strategies,
            missing_feature_config,
        )
    except ValueError as error:
        assert "does_not_exist" in str(error)
    else:
        raise AssertionError(
            "Missing context metadata was not rejected"
        )

    print(
        "\nComponent 6 conditioning-registry test passed."
    )

def inspect_actor_input_assembly(
    world: World,
) -> None:
    """Verify observation/context concatenation for actor inputs."""

    algorithm = world.learning_role.rl_algorithm

    observation = th.arange(
        algorithm.obs_dim,
        dtype=algorithm.float_type,
        device=algorithm.device,
    )

    original_observation = observation.clone()

    for unit_id in world.learning_role.rl_strats:
        context = algorithm.conditioning_tensor(
            unit_id
        )

        actor_input = algorithm.prepare_actor_input(
            unit_id,
            observation,
        )

        assert actor_input.shape == (
            algorithm.actor_input_dim,
        )

        assert th.equal(
            actor_input[:algorithm.obs_dim],
            observation,
        )

        assert th.equal(
            actor_input[algorithm.obs_dim:],
            context,
        )

        batch = observation.unsqueeze(0).repeat(
            3,
            1,
        )

        batched_actor_input = (
            algorithm.prepare_actor_input(
                unit_id,
                batch,
            )
        )

        assert batched_actor_input.shape == (
            3,
            algorithm.actor_input_dim,
        )

        expected_context_batch = (
            algorithm.conditioning_tensor(
                unit_id,
                batch_size=3,
            )
        )

        assert th.equal(
            batched_actor_input[
                :,
                algorithm.obs_dim:
            ],
            expected_context_batch,
        )

    # Input construction must not mutate the original observation.
    assert th.equal(
        observation,
        original_observation,
    )

    try:
        algorithm.prepare_actor_input(
            next(iter(world.learning_role.rl_strats)),
            th.zeros(
                2,
                3,
                algorithm.obs_dim,
                dtype=algorithm.float_type,
                device=algorithm.device,
            ),
        )
    except ValueError as error:
        assert "one-dimensional or two-dimensional" in str(
            error
        )
    else:
        raise AssertionError(
            "Three-dimensional actor input was not rejected"
        )

    print(
        "\nComponent 7.1 actor-input assembly test passed."
    )

def inspect_contextual_actor_forward(
    world: World,
) -> None:
    """Verify actors are built for and accept contextual inputs."""

    algorithm = world.learning_role.rl_algorithm

    for unit_id, strategy in (
        world.learning_role.rl_strats.items()
    ):
        observation = th.zeros(
            algorithm.obs_dim,
            dtype=algorithm.float_type,
            device=algorithm.device,
        )

        actor_input = algorithm.prepare_actor_input(
            unit_id,
            observation,
        )

        assert strategy.actor.FC1.in_features == (
            algorithm.actor_input_dim
        )

        with th.no_grad():
            if algorithm.learning_config.algorithm == "mappo":
                action = strategy.actor(
                    actor_input,
                    deterministic=True,
                )
            else:
                action = strategy.actor(actor_input)

        assert action.shape == (
            algorithm.act_dim,
        )

    print(
        "\nComponent 7.2 contextual actor forward test passed."
    )

def inspect_standalone_contextual_inference(
    world: World,
) -> None:
    """Verify contextual actors survive standalone loading."""

    algorithm = world.learning_role.rl_algorithm
    unit_id, strategy = next(
        iter(world.learning_role.rl_strats.items())
    )

    observation = th.zeros(
        algorithm.obs_dim,
        dtype=algorithm.float_type,
        device=algorithm.device,
    )

    original_actor = strategy.actor
    original_learning_mode = strategy.learning_mode
    original_evaluation_mode = strategy.evaluation_mode
    original_conditioning = getattr(
        strategy,
        "actor_conditioning",
        None,
    )
    original_feature_names = getattr(
        strategy,
        "actor_conditioning_feature_names",
        (),
    )

    contextual_input = algorithm.prepare_actor_input(
        unit_id,
        observation,
    )

    with th.no_grad():
        if algorithm.learning_config.algorithm == "mappo":
            expected_action = original_actor(
                contextual_input,
                deterministic=True,
            )
        else:
            expected_action = original_actor(
                contextual_input
            )

    try:
        with TemporaryDirectory(
            prefix="assume-standalone-context-"
        ) as temporary_directory:
            actors_directory = (
                Path(temporary_directory) / "actors"
            )

            algorithm.save_actor_params(
                directory=str(actors_directory)
            )

            strategy.load_actor_params(
                load_path=temporary_directory
            )

            strategy.learning_mode = False
            strategy.evaluation_mode = False

            actual_action, noise = strategy.get_actions(
                observation
            )

            assert strategy.actor.FC1.in_features == (
                algorithm.actor_input_dim
            )

            assert strategy.actor_conditioning.numel() == (
                algorithm.conditioning_registry.context_dim
            )

            assert th.allclose(
                actual_action,
                expected_action,
            )

            assert th.count_nonzero(noise).item() == 0

    finally:
        strategy.actor = original_actor
        strategy.learning_mode = original_learning_mode
        strategy.evaluation_mode = original_evaluation_mode
        strategy.actor_conditioning = original_conditioning
        strategy.actor_conditioning_feature_names = (
            original_feature_names
        )

    print(
        "\nComponent 7.3 standalone contextual "
        "inference test passed."
    )

def load_learning_results(
    database_path: Path,
    simulation_id: str,
) -> pd.DataFrame:
    """Load training data for one isolated experiment."""

    engine = create_engine(
        f"sqlite:///{database_path}"
    )

    query = """
    SELECT
        episode,
        unit,
        reward,
        profit,
        regret,
        actions_0,
        actions_1,
        evaluation_mode
    FROM rl_params
    WHERE simulation = :simulation_id
    ORDER BY episode, datetime, unit
    """

    return pd.read_sql_query(
        query,
        engine,
        params={"simulation_id": simulation_id},
    )

def load_final_market_prices(
    database_path: Path,
    simulation_id: str,
) -> pd.Series:
    """Load the final trained-policy market-clearing prices."""

    engine = create_engine(
        f"sqlite:///{database_path}"
    )

    query = """
    SELECT
        product_start AS datetime,
        price
    FROM market_meta
    WHERE simulation = :simulation_id
      AND market_id = 'EOM'
      AND episode is NULL
    ORDER BY product_start
    """

    frame = pd.read_sql_query(
        query,
        engine,
        params={"simulation_id": simulation_id},
    )

    if frame.empty:
        raise ValueError(
            f"No final market prices found for {simulation_id}"
        )

    if frame["datetime"].duplicated().any():
        duplicate_datetimes = frame.loc[
            frame["datetime"].duplicated(keep=False),
            "datetime",
        ].astype(str).unique().tolist()

        raise ValueError(f"Final market-price query returned multiple prices for the same timestamp. Duplicate timestamps: {duplicate_datetimes[:10]}")

    frame["datetime"] = pd.to_datetime(
        frame["datetime"]
    )

    return pd.Series(
        frame["price"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["datetime"]),
        name="simulation_price",
    )

def load_reference_prices(
    example_name: str,
) -> pd.Series:
    """Load and hourly-resample the reference price series."""

    path = (
        REFERENCE_PRICE_DIRECTORY
        / f"prices_{example_name.removeprefix('example_')}.csv"
    )

    frame = pd.read_csv(
        path,
        sep=";",
    )

    frame["datetime"] = pd.to_datetime(
        frame["datetime"],
        dayfirst=True,
    )

    frame["equilibirum price"] = pd.to_numeric(
        frame["equilibirum price"],
        errors="coerce",
    )

    series = (
        frame
        .set_index("datetime")["equilibirum price"]
        .dropna()
        .sort_index()
        .resample("1h")
        .mean()
    )

    return series.rename("reference_price")

def calculate_price_metrics(
    simulation_prices: pd.Series,
    reference_prices: pd.Series,
) -> dict[str, float]:
    """Compare simulated prices with aligned reference prices."""

    aligned = pd.concat(
        [
            simulation_prices.rename("simulation"),
            reference_prices.rename("reference"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError(
            "Simulation and reference prices have no "
            "overlapping timestamps"
        )

    error = (
        aligned["simulation"]
        - aligned["reference"]
    )

    correlation = aligned[
        ["simulation", "reference"]
    ].corr().iloc[0, 1]

    return {
        "n_prices": int(len(aligned)),
        "price_mae": float(error.abs().mean()),
        "price_rmse": float(
            np.sqrt(np.mean(np.square(error)))
        ),
        "price_bias": float(error.mean()),
        "price_correlation": float(correlation),
    }

def calculate_reward_curve(
    results: pd.DataFrame,
) -> pd.Series:
    """Calculate mean episode return across learning units."""

    training = results[
        results["evaluation_mode"] == 0
    ]

    unit_returns = (
        training
        .groupby(["episode", "unit"])["reward"]
        .sum()
    )

    return (
        unit_returns
        .groupby("episode")
        .mean()
        .sort_index()
    )

def calculate_profit_curve(
    results: pd.DataFrame,
) -> pd.Series:
    training = results[
        results["evaluation_mode"] == 0
    ]

    unit_profit = (
        training
        .groupby(["episode", "unit"])["profit"]
        .sum()
    )

    return (
        unit_profit
        .groupby("episode")
        .mean()
        .sort_index()
    )

def run_one_experiment(
    spec: ExperimentSpec,
) -> dict:
    """Train and evaluate one isolated algorithm/configuration."""

    run_directory = (
        EXPERIMENT_DIRECTORY / spec.run_id
    )
    database_path = run_directory / "results.db"
    policy_directory = run_directory / "policies"
    csv_directory = run_directory / "simulation"

    if run_directory.exists():
        shutil.rmtree(run_directory)

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix=f"{spec.run_id}-"
    ) as temporary_directory:
        inputs_path, scenario_name = (
            configure_temporary_scenario(
                spec=spec,
                temporary_root=Path(temporary_directory),
                policy_directory=policy_directory,
            )
        )

        world = World(
            database_uri=(
                f"sqlite:///{database_path}"
            ),
            export_csv_path=str(csv_directory),
        )

        world.bidding_strategies[
            "powerplant_energy_learning"
        ] = EnergyLearningStrategy

        load_scenario_folder(
            world=world,
            inputs_path=str(inputs_path),
            scenario=scenario_name,
            study_case=STUDY_CASE,
        )

        start_time = time.perf_counter()

        run_learning(
            world=world,
            verbose=False,
        )

        # Execute the final trained-policy evaluation.
        world.run()

        runtime_seconds = (
            time.perf_counter() - start_time
        )

        simulation_id = (
            f"{spec.example_name}_{STUDY_CASE}"
        )

        training_results = load_learning_results(
            database_path,
            simulation_id,
        )

        price_series = load_final_market_prices(
            database_path,
            simulation_id,
        )

    reference_prices = load_reference_prices(
        spec.example_name
    )

    price_metrics = calculate_price_metrics(
        price_series,
        reference_prices,
    )

    result = {
        "spec": spec,
        "reward_curve": calculate_reward_curve(
            training_results
        ),
        "profit_curve": calculate_profit_curve(
            training_results
        ),
        "market_prices": price_series,
        "reference_prices": reference_prices,
        "runtime_seconds": runtime_seconds,
        **price_metrics,
    }

    save_experiment_result(
        result,
        run_directory,
    )

    return result

def save_experiment_result(
    result: dict,
    run_directory: Path,
) -> None:
    result["reward_curve"].rename(
        "reward"
    ).to_csv(run_directory / "reward_curve.csv")

    result["profit_curve"].rename(
        "profit"
    ).to_csv(run_directory / "profit_curve.csv")

    result["market_prices"].rename(
        "price"
    ).to_csv(run_directory / "market_prices.csv")

    metrics = {
        key: value
        for key, value in result.items()
        if key in {
            "runtime_seconds",
            "n_prices",
            "price_mae",
            "price_rmse",
            "price_bias",
            "price_correlation",
        }
    }

    with (
        run_directory / "metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

def plot_parameter_sharing_comparison(
    results: list[dict],
    example_name: str,
) -> None:
    """Plot before/after results for all three algorithms."""

    figure, axes = plt.subplots(
        nrows=len(ALGORITHMS),
        ncols=3,
        figsize=(20, 13),
        constrained_layout=True,
    )

    colors = {
        "independent": "black",
        "full_none": "tab:blue",
        "full_one_hot": "tab:orange",
        "full_semantic": "tab:green",
        "full_id_context": "tab:red",
    }

    for row, algorithm in enumerate(ALGORITHMS):
        algorithm_results = [
            result
            for result in results
            if (
                result["spec"].example_name
                == example_name
                and result["spec"].algorithm
                == algorithm
            )
        ]

        reward_axis = axes[row, 0]
        price_axis = axes[row, 1]
        metric_axis = axes[row, 2]

        for result in algorithm_results:
            variant = result["spec"].variant_name
            reward_curve = result["reward_curve"]

            reward_axis.plot(
                reward_curve.index,
                reward_curve.values,
                label=variant,
                color=colors[variant],
            )

        reward_axis.set_title(
            f"{algorithm.upper()} — training return"
        )
        reward_axis.set_xlabel("Episode")
        reward_axis.set_ylabel("Mean return per unit")
        reward_axis.grid(alpha=0.3)

        if algorithm_results:
            reference_prices = algorithm_results[
                0
            ]["reference_prices"]

            # Plot the first week for readability.
            reference_window = reference_prices.loc[
                reference_prices.index.min():
                reference_prices.index.min()
                + pd.Timedelta(days=7)
            ]

            price_axis.step(
                reference_window.index,
                reference_window.values,
                where="post",
                color="black",
                linewidth=2,
                label="reference prices",
            )

        for result in algorithm_results:
            variant = result["spec"].variant_name
            prices = result["market_prices"]

            if algorithm_results:
                start = reference_window.index.min()
                end = reference_window.index.max()
                prices = prices.loc[start:end]

            price_axis.step(
                prices.index,
                prices.values,
                where="post",
                color=colors[variant],
                alpha=0.8,
                label=variant,
            )

        price_axis.set_title(
            f"{algorithm.upper()} — final prices"
        )
        price_axis.set_ylabel("EUR/MWh")
        price_axis.grid(alpha=0.3)

        variant_names = [
            result["spec"].variant_name
            for result in algorithm_results
        ]
        maes = [
            result["price_mae"]
            for result in algorithm_results
        ]

        metric_axis.bar(
            variant_names,
            maes,
            color=[
                colors[name]
                for name in variant_names
            ],
        )
        metric_axis.set_title(
            f"{algorithm.upper()} — price MAE"
        )
        metric_axis.set_ylabel("MAE [EUR/MWh]")
        metric_axis.tick_params(
            axis="x",
            rotation=35,
        )
        metric_axis.grid(
            axis="y",
            alpha=0.3,
        )

        reward_axis.legend(fontsize=8)
        price_axis.legend(fontsize=8)

    output_path = (
        EXPERIMENT_DIRECTORY
        / f"{example_name}_before_after.png"
    )

    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)

    print(f"Saved comparison plot: {output_path}")

def parse_arguments() -> argparse.Namespace:
    """Parse paths that may differ between local environments."""

    parser = argparse.ArgumentParser(
        description="Run the parameter-sharing experiment matrix.",
    )
    parser.add_argument(
        "--inputs-directory",
        type=Path,
        default=INPUTS_PATH,
        help=(
            "Directory containing ASSUME scenario folders "
            f"(default: {INPUTS_PATH})"
        ),
    )
    parser.add_argument(
        "--reference-price-directory",
        type=Path,
        default=REFERENCE_PRICE_DIRECTORY,
        help=(
            "Directory containing prices_<scenario>.csv files "
            f"(default: {REFERENCE_PRICE_DIRECTORY})"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help=(
            "Directory for databases, policies, and plots "
            f"(default: {OUTPUT_DIRECTORY})"
        ),
    )

    return parser.parse_args()

def configure_paths(arguments: argparse.Namespace) -> None:
    """Resolve and validate filesystem paths used by the experiment."""

    global INPUTS_PATH
    global OUTPUT_DIRECTORY
    global DATABASE_DIRECTORY
    global REFERENCE_PRICE_DIRECTORY
    global EXPERIMENT_DIRECTORY

    INPUTS_PATH = (
        arguments.inputs_directory.expanduser().resolve()
    )
    REFERENCE_PRICE_DIRECTORY = (
        arguments.reference_price_directory.expanduser().resolve()
    )
    OUTPUT_DIRECTORY = (
        arguments.output_directory.expanduser().resolve()
    )
    DATABASE_DIRECTORY = OUTPUT_DIRECTORY / "local_db"
    EXPERIMENT_DIRECTORY = (
        OUTPUT_DIRECTORY / "parameter_sharing_experiments"
    )

    if not INPUTS_PATH.is_dir():
        raise FileNotFoundError(
            f"ASSUME input directory does not exist: {INPUTS_PATH}"
        )

    if not REFERENCE_PRICE_DIRECTORY.is_dir():
        raise FileNotFoundError(
            "Reference-price directory does not exist: "
            f"{REFERENCE_PRICE_DIRECTORY}"
        )

def main() -> None:
    configure_paths(parse_arguments())

    variants = tuple(SHARING_VARIANTS)

    specifications = [
        ExperimentSpec(
            example_name=example_name,
            algorithm=algorithm,
            variant_name=variant,
            seed=seed,
            training_episodes=50,
        )
        for example_name in EXAMPLES
        for algorithm in ALGORITHMS
        for variant in variants
        for seed in SEEDS
    ]

    results = []

    for index, specification in enumerate(
        specifications,
        start=1,
    ):
        print(
            f"\n[{index}/{len(specifications)}] "
            f"{specification.run_id}"
        )

        results.append(
            run_one_experiment(specification)
        )

    plot_parameter_sharing_comparison(
        results,
        example_name="example_02b",
    )

# def main() -> None:
#     """Run the complete native ASSUME learning pipeline."""

#     world = create_world()
#     load_example(world)
#     # print_parameter_sharing_config(world)
#     inspect_group_registries(world)
#     inspect_conditioning_registry(world)
#     inspect_actor_ownership(world)
#     inspect_actor_input_assembly(world)
#     inspect_contextual_actor_forward(world)
#     inspect_standalone_contextual_inference(world)
#     inspect_actor_loss_aggregation(world)
#     inspect_actor_checkpoint_roundtrip(world)

#     # sharing_config = world.learning_role.learning_config.parameter_sharing

#     # has_real_sharing = any(
#     #     len(members) > 1
#     #     for members in (
#     #         world.learning_role
#     #         .rl_algorithm
#     #         .actor_group_registry
#     #         .groups
#     #         .values()
#     #     )
#     # )

#     # if sharing_config.enabled and has_real_sharing:
#     #     print(
#     #         "\nComponent 3 ownership test passed. "
#     #         "Training is intentionally skipped until Component 4 "
#     #         "implements one optimizer update per actor group."
#     #     )
#     #     return

#     # ASSUME performs policy initialization, buffer creation, training,
#     # validation, checkpointing, and final evaluation setup internally.
#     run_learning(world=world, verbose=False)

#     world.run()

#     results = load_learning_results()

#     max_bid_price = (
#         world.learning_role.learning_config.max_bid_price
#     )

#     episode_returns, bid_prices = prepare_training_curves(
#         results=results,
#         max_bid_price=max_bid_price,
#     )

#     # All learning units in example_02b are identical CCGTs, so one reference
#     # marginal cost is sufficient.
#     learning_units = [
#         unit
#         for operator in world.unit_operators.values()
#         for unit in operator.units.values()
#         if any(
#             isinstance(strategy, EnergyLearningStrategy)
#             for strategy in unit.bidding_strategies.values()
#         )
#     ]

#     reference_unit = learning_units[0]
#     reference_time = reference_unit.index[0]

#     marginal_cost = reference_unit.calculate_marginal_cost(
#         start=reference_time,
#         power=reference_unit.max_power,
#     )

#     print(f"Collected training episodes: {len(episode_returns)}")
#     print(f"Learning units: {list(bid_prices.columns)}")

#     plot_results(
#         episode_returns=episode_returns,
#         bid_prices=bid_prices,
#         marginal_cost=marginal_cost,
#     )

if __name__ == "__main__":
    main()
