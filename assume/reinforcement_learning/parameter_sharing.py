# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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