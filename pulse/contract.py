"""Validates one MBTA /predictions payload page against
contracts/mbta-predictions.v1.json.

M2 must-address item 4. Hand-rolled, not the `jsonschema` library: the
allowed dependency list for this project is scikit-learn/mlflow/joblib
(+pandas, optional) -- adding a schema-validation library for a contract
this small and this specific to one endpoint's shape isn't worth a new
dependency, and a bespoke ~80-line validator against a bespoke, deliberately
narrow contract format (types/required/nullable/allowed-values -- see the
contract file's own docstring for why it only covers the fields map_rows
actually reads) is easier to read end-to-end than a generic JSON-Schema
validator would be here.

validate_page returns a list of violation strings (empty when the page is
clean) rather than raising -- pulse.mbta.fetch_predictions calls this once
per raw page (before merging pages together, so this really does see one
HTTP response's worth of data, not a whole batch) and folds any violations
into what it returns; pulse.poll is the layer that turns a non-empty list
into a named poll_runs error. A violating page's data is still dropped
before merging (see fetch_predictions) -- fail loud in the ledger, but keep
the cycle's other, valid pages.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONTRACT_PATH = Path(__file__).resolve().parent.parent / "contracts" / "mbta-predictions.v1.json"

_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    return json.loads(path.read_text())


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "string_datetime":
        return isinstance(value, str) and bool(_DATETIME_RE.match(value))
    if expected_type == "string_date":
        return isinstance(value, str) and bool(_DATE_RE.match(value))
    raise ValueError(f"pulse.contract: unknown contract type {expected_type!r}")


def _validate_attributes(attrs: Mapping[str, Any], spec: Mapping[str, Any], where: str) -> list[str]:
    violations = []
    for name, attr_spec in spec.items():
        present = name in attrs
        if not present:
            if attr_spec.get("required"):
                violations.append(f"{where}.attributes.{name}: missing (required by contract)")
            continue
        value = attrs[name]
        if value is None:
            if not attr_spec.get("nullable", False):
                violations.append(f"{where}.attributes.{name}: null, but contract marks this field non-nullable")
            continue
        if not _matches_type(value, attr_spec["type"]):
            violations.append(
                f"{where}.attributes.{name}: value {value!r} does not match contract type {attr_spec['type']!r}"
            )
            continue
        allowed = attr_spec.get("allowed_values")
        if allowed is not None and value not in allowed:
            violations.append(f"{where}.attributes.{name}: value {value!r} not in allowed values {allowed!r}")
    return violations


def _validate_relationships(rels: Mapping[str, Any], spec: Mapping[str, Any], where: str) -> list[str]:
    violations = []
    for name, rel_spec in spec.items():
        rel = rels.get(name)
        if rel is None:
            if rel_spec.get("required"):
                violations.append(f"{where}.relationships.{name}: missing (required by contract)")
            continue
        data = rel.get("data")
        if data is None:
            if not rel_spec.get("nullable_data", False):
                violations.append(
                    f"{where}.relationships.{name}.data: null, but contract marks this relationship non-nullable"
                )
            continue
        rid = data.get("id") if isinstance(data, Mapping) else None
        if not isinstance(rid, str):
            violations.append(f"{where}.relationships.{name}.data.id: not a string ({rid!r})")
    return violations


def _validate_resource(resource: Mapping[str, Any], spec: Mapping[str, Any], where: str) -> list[str]:
    violations = []
    for field in spec.get("required_top_level_fields", []):
        if field not in resource:
            violations.append(f"{where}: missing required top-level field {field!r}")
    attrs = resource.get("attributes") or {}
    violations.extend(_validate_attributes(attrs, spec.get("attributes", {}), where))
    if "relationships" in spec:
        rels = resource.get("relationships") or {}
        violations.extend(_validate_relationships(rels, spec["relationships"], where))
    return violations


def validate_page(payload: Mapping[str, Any], contract: Mapping[str, Any] | None = None) -> list[str]:
    """Validate one payload page's `data` (prediction resources) and
    `included` (only `type == "schedule"` resources -- see the contract
    file's docstring for why route/stop/trip/vehicle resources in
    `included` aren't checked). Pure -- no I/O beyond the one-time contract
    load when `contract` isn't passed in."""
    contract = contract if contract is not None else load_contract()
    resource_types = contract["resource_types"]
    violations: list[str] = []

    for i, pred in enumerate(payload.get("data") or []):
        rtype = pred.get("type")
        if rtype != "prediction":
            violations.append(f"data[{i}].type: {rtype!r}, expected 'prediction'")
            continue
        violations.extend(_validate_resource(pred, resource_types["prediction"], f"data[{i}]"))

    for i, item in enumerate(payload.get("included") or []):
        if item.get("type") != "schedule":
            continue
        violations.extend(_validate_resource(item, resource_types["schedule"], f"included[{i}](schedule)"))

    return violations
