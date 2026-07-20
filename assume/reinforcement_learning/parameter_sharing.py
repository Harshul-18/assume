# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from numbers import Real
from dataclasses import dataclass
from typing import Mapping

from assume.common.base import (
    LearningStrategy,
    ParameterSharingConfig
)

# PARAMETER-SHARING
# component 2: grouping

@dataclass
class NetworkGroupRegistry:
    """Store both directions of a network-sharing assignment."""
    unit_to_group: dict[str, str]
    groups: dict[str, tuple[str, ...]]

    def group_for(self, unit_id: str) -> str:
        """return the group assigned to one unit."""
        try:
            return self.unit_to_group[unit_id]
        except KeyError as error:
            raise KeyError(
                f"Unit '{unit_id}' has no parameter-sharing group."
            ) from error

# PARAMETER-SHARING
# component 6: conditioning-vector construction
@dataclass
class ConditioningRegistry:
    """Store the fixed conditioning vector for every learning unit."""
    
    unit_to_vector: dict[str, tuple[float, ...]]
    feature_names: tuple[str, ...]
    numeric_bounds: dict[str, tuple[float, float]]
    categorical_values: dict[str, tuple[str, ...]]

    @property
    def context_dim(self) -> int:
        """Return the number of values appended to an observation."""
        return len(self.feature_names)
    
    def vector_for(
        self,
        unit_id: str,
    ) -> tuple[float, ...]:
        """Return the conditioning vector assigned to one unit."""
        try:
            return self.unit_to_vector[unit_id]
        except KeyError as error:
            raise KeyError(f"Unit '{unit_id}' has no conditioning vector") from error

def build_group_registry(
    strategies: Mapping[str, LearningStrategy],
    mode: str,
    config: ParameterSharingConfig,
    network_name: str,
) -> NetworkGroupRegistry:
    """Build actor or critic groups without creating networks."""
    unit_ids = list(strategies.keys())
    if not unit_ids:
        raise ValueError(
            f"Cannot create {network_name} groups without learning units"
        )
    
    # preserving the original independent topology if parameter sharing is not enabled or the mode is set to be independent.
    if not config.enabled or mode == "independent":
        mapping = {
            unit_id: f"unit:{unit_id}"
            for unit_id in unit_ids
        }
    # these modes conceptually use one shared network or shared trunk.
    elif mode in {"full", "shared_trunk", "context_conditioned"}:
        mapping = {
            unit_id: "all"
            for unit_id in unit_ids
        }
    elif mode == "grouped":
        mapping = _build_grouped_mapping(
            strategies = strategies,
            config = config
        )
    else:
        raise ValueError(
            f"Unsupported {network_name} sharing mode: '{mode}'"
        )

    groups = _invert_mapping(
        unit_ids = unit_ids,
        unit_to_group = mapping
    )

    _validate_group_sizes(
        groups = groups,
        max_group_size = config.max_group_size,
        network_name = network_name
    )

    return NetworkGroupRegistry(
        unit_to_group = mapping,
        groups = groups
    )

def _build_grouped_mapping(
    strategies: Mapping[str, LearningStrategy],
    config: ParameterSharingConfig
) -> dict[str, str]:
    """Apply the configured grouped method."""
    unit_ids = list(strategies.keys())
    method = config.grouping_method

    if method == "individual":
        return {
            unit_id: f"unit:{unit_id}"
            for unit_id in unit_ids
        }
    if method == "all":
        return {
            unit_id: "all"
            for unit_id in unit_ids
        }
    if method == "manual":
        return _build_manual_mapping(
            unit_ids = unit_ids,
            configured_mapping = config.unit_to_group
        )
    if method == "semantic":
        return _build_semantic_mapping(
            strategies = strategies,
            grouping_feature = config.grouping_feature
        )
    if method in {
        "kmeans_context",
        "learned_embedding",
        "dynamic_quantile"
    }:
        raise NotImplementedError(
            f"Grouping method '{method}' is planned, but not implemented yet."
        )
    raise ValueError(f"Unknown grouping method: '{method}'")

def _build_manual_mapping(
    unit_ids: list[str],
    configured_mapping: dict[str, str]
) -> dict[str, str]:
    """Validate and return an explicit user-provided mapping."""
    expected_units = set(unit_ids)
    configured_units = set(configured_mapping)

    missing_units = expected_units - configured_units
    unknown_units = configured_units - expected_units

    if missing_units:
        raise ValueError(
            f"Manual parameter-sharing mapping is missing units: {sorted(missing_units)}"
        )
    if unknown_units:
        raise ValueError(
            f"Manual parameter-sharing mapping contains unkown units: {sorted(unknown_units)}"
        )

    mapping = {}

    for unit_id in unit_ids:
        group_id = configured_mapping[unit_id]
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(
                f"Unit '{unit_id}' has an invalid group ID: {group_id!r}"
            )
        mapping[unit_id] = group_id.strip()
    
    return mapping

def _build_semantic_mapping(
    strategies: Mapping[str, LearningStrategy],
    grouping_feature: str
) -> dict[str, str]:
    """Group units using metadata such as technology."""
    mapping = {}

    for unit_id, strategy in strategies.items():
        metadata = getattr(strategy, "sharing_metadata", {})
        value = metadata.get(grouping_feature)
        if value is None or str(value).strip() == "":
            available_features = sorted(metadata.keys())

            raise ValueError(
                f"Unit '{unit_id}' does not provide semantic grouping feature '{grouping_feature}'. Available features for this unit: {available_features}"
            )
        mapping[unit_id] = f"{grouping_feature}:{str(value).strip()}"

    return mapping

def _invert_mapping(
    unit_ids: list[str],
    unit_to_group: dict[str, str]
) -> dict[str, tuple[str, ...]]:
    """create group-to-members mapping while preserving unit order."""
    mutable_groups: dict[str, list[str]] = {}

    for unit_id in unit_ids:
        group_id = unit_to_group[unit_id]
        mutable_groups.setdefault(group_id, []).append(unit_id)
    
    return {
        group_id: tuple(member_ids)
        for group_id, member_ids in mutable_groups.items()
    }

def _validate_group_sizes(
    groups: dict[str, tuple[str, ...]],
    max_group_size: int | None,
    network_name: str,
) -> None:
    """Reject oversized groups until automatic splitting is implemented."""

    if max_group_size is None:
        return

    oversized = {
        group_id: len(members)
        for group_id, members in groups.items()
        if len(members) > max_group_size
    }

    if oversized:
        raise ValueError(
            f"{network_name.capitalize()} groups exceed max_group_size={max_group_size}: {oversized}"
        )

# PARAMETER-SHARING
# component 6: conditioning-vector construction
def build_conditioning_registry(
    strategies: Mapping[str, LearningStrategy],
    config: ParameterSharingConfig
) -> ConditioningRegistry:
    """Building fixed identity/context vectors for all learning units."""
    unit_ids = tuple(strategies)
    if not unit_ids:
        raise ValueError("Cannot construct conditioning vectors without learning units.")
    method = config.conditioning_method
    if not config.enabled or method == "none":
        return _build_empty_conditioning(unit_ids)
    if method == "one_hot_unit_id":
        return _build_one_hot_conditioning(unit_ids)
    if method == "semantic_context":
        return _build_semantic_conditioning(
            strategies = strategies,
            context_features = config.context_features
        )
    if method == "unit_id_and_context":
        identity_registry = _build_one_hot_conditioning(unit_ids)
        semantic_registry = _build_semantic_conditioning(strategies, config.context_features)
        return _combine_conditioning_registries(identity_registry, semantic_registry)
    if method == "learned_embedding":
        raise NotImplementedError("")
    raise ValueError(f"Unsupported conditioning method: '{method}'")

# PARAMETER-SHARING
# component 6: conditioning-vector construction
def _build_empty_conditioning(unit_ids: tuple[str, ...]) -> ConditioningRegistry:
    """Represent every unit using an empty context vector."""
    return ConditioningRegistry(
        unit_to_vector = {unit_id: () for unit_id in unit_ids},
        feature_names = (),
        numeric_bounds = {},
        categorical_values = {}
    )

# PARAMETER-SHARING
# component 6: conditioning-vector construction
def _build_one_hot_conditioning(unit_ids: tuple[str, ...]) -> ConditioningRegistry:
    """Assign a deterministic one-hot identity to every unit."""
    feature_names = tuple(f"unit_id={unit_id}" for unit_id in unit_ids)
    unit_to_vector = {}
    for current_index, unit_id in enumerate(unit_ids):
        vector = tuple(
            1.0 if feature_index == current_index else 0.0
            for feature_index in range(len(unit_ids))
        )
        unit_to_vector[unit_id] = vector
    return ConditioningRegistry(
        unit_to_vector = unit_to_vector,
        feature_names = feature_names,
        numeric_bounds = {},
        categorical_values = {}
    )

# PARAMETER-SHARING
# component 6: conditioning-vector construction
def _build_semantic_conditioning(
    strategies: Mapping[str, LearningStrategy],
    context_features: list[str]
) -> ConditioningRegistry:
    """Encode configured semantic metadata as numerical vectors."""
    if not context_features:
        raise ValueError("Semantic conditioning requires at least one context feature.")
    unit_ids = tuple(strategies)
    vector_parts: dict[str, list[float]] = {unit_id: [] for unit_id in unit_ids}
    expanded_feature_names: list[str] = []
    numeric_bounds: dict[str, tuple[float, float]] = {}
    categorical_values: dict[str, tuple[str, ...]] = {}
    for feature_name in context_features:
        values_by_unit = {}
        for unit_id, strategy in strategies.items():
            metadata = getattr(strategy, "sharing_metadata", {})
            if feature_name not in metadata:
                raise ValueError(f"Unit '{unit_id}' does not provide context feature '{feature_name}'. Available features: {sorted(metadata)}.")
            value = metadata[feature_name]
            if value is None:
                raise ValueError(f"Unit '{unit_id}' provides None for context feature '{feature_name}'")
            values_by_unit[unit_id] = value
        values = tuple(values_by_unit.values())
        numeric_flags = tuple(isinstance(value, Real) and not isinstance(value, bool) for value in values)
        categorical_flags = tuple(isinstance(value, (str, bool)) for value in values)
        if all(numeric_flags):
            numeric_values = {unit_id: float(value) for unit_id, value in values_by_unit.items()}
            non_finite_units = [unit_id for unit_id, value in numeric_values.items() if not math.isfinite(value)]
            if non_finite_units:
                raise ValueError(f"Feature '{feature_name}' must be finite for all units. Found non-finite values in: {non_finite_units}")
            lower = min(numeric_values.values())
            upper = max(numeric_values.values())
            numeric_bounds[feature_name] = (lower, upper)
            expanded_feature_names.append(feature_name)
            for unit_id, value in numeric_values.items():
                if upper == lower:
                    normalized_value = 0.0
                else:
                    normalized_value = (value - lower) / (upper - lower)
                vector_parts[unit_id].append(normalized_value)
            continue
        if all(categorical_flags):
            string_values = {unit_id: str(value) for unit_id, value in values_by_unit.items()}
            categories = tuple(sorted(set(string_values.values())))
            categorical_values[feature_name] = categories
            expanded_feature_names.extend(f"{feature_name}={category}" for category in categories)
            for unit_id, value in string_values.items():
                vector_parts[unit_id].extend(1.0 if value == category else 0.0 for category in categories)
            continue
        value_types = {unit_id: type(value).__name__ for unit_id, value in values_by_unit.items()}
        raise TypeError(f"Context feature '{feature_name}' mixes unsupported types: {value_types}")
    unit_to_vector = {unit_id: tuple(values) for unit_id, values in vector_parts.items()}
    return ConditioningRegistry(
        unit_to_vector = unit_to_vector,
        feature_names = tuple(expanded_feature_names),
        numeric_bounds = numeric_bounds,
        categorical_values = categorical_values
    )

# PARAMETER-SHARING
# component 6: conditioning-vector construction
def _combine_conditioning_registries(
    first: ConditioningRegistry,
    second: ConditioningRegistry,
) -> ConditioningRegistry:
    """Concatenate two compatible conditioning registries."""
    if set(first.unit_to_vector) != set(second.unit_to_vector):
        raise ValueError("Cannot combine conditioning registries with different units")
    unit_to_vector = {
        unit_id: (first.vector_for(unit_id) + second.vector_for(unit_id))
        for unit_id in first.unit_to_vector
    }
    overlapping_numeric = (set(first.numeric_bounds) & set(second.numeric_bounds))
    overlapping_categorical = (set(first.categorical_values) & set(second.categorical_values))
    if overlapping_numeric or overlapping_categorical:
        raise ValueError("Cannot combine conditioning registries with overlapping metadata definitions")
    return ConditioningRegistry(
        unit_to_vector=unit_to_vector,
        feature_names=(first.feature_names + second.feature_names),
        numeric_bounds={
            **first.numeric_bounds,
            **second.numeric_bounds,
        },
        categorical_values={
            **first.categorical_values,
            **second.categorical_values,
        },
    )