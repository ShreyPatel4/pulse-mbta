"""Tests for pulse.contract: pure validation of a /predictions payload page
against contracts/mbta-predictions.v1.json. Pure -- no I/O beyond the
one-time contract load in the module-level FIXTURE/CONTRACT below."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from pulse import contract

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE = json.loads((FIXTURES_DIR / "predictions_sample.json").read_text())
CONTRACT = contract.load_contract()


def _mutated(mutate) -> dict:
    """Deep-copy the real fixture and apply `mutate` to it -- every test
    below mutates a real, previously-observed-good payload rather than
    hand-building a synthetic one, so a violation is only ever introduced by
    the specific mutation under test."""
    payload = copy.deepcopy(FIXTURE)
    mutate(payload)
    return payload


def test_load_contract_has_the_two_resource_types_map_rows_reads():
    assert set(CONTRACT["resource_types"]) == {"prediction", "schedule"}


def test_real_fixture_validates_clean():
    assert contract.validate_page(FIXTURE, CONTRACT) == []


def test_empty_payload_validates_clean():
    assert contract.validate_page({"data": [], "included": []}, CONTRACT) == []


def test_missing_required_attribute_is_a_violation():
    def mutate(payload):
        del payload["data"][0]["attributes"]["direction_id"]

    violations = contract.validate_page(_mutated(mutate), CONTRACT)
    assert len(violations) == 1
    assert "attributes.direction_id" in violations[0]
    assert "missing" in violations[0]


def test_wrong_type_attribute_is_a_violation():
    def mutate(payload):
        payload["data"][0]["attributes"]["direction_id"] = "0"  # string, not int

    violations = contract.validate_page(_mutated(mutate), CONTRACT)
    assert len(violations) == 1
    assert "attributes.direction_id" in violations[0]
    assert "contract type" in violations[0]


def test_out_of_range_allowed_value_is_a_violation():
    def mutate(payload):
        payload["data"][0]["attributes"]["direction_id"] = 2

    violations = contract.validate_page(_mutated(mutate), CONTRACT)
    assert len(violations) == 1
    assert "not in allowed values" in violations[0]


def test_null_direction_id_is_a_violation_not_nullable():
    def mutate(payload):
        payload["data"][0]["attributes"]["direction_id"] = None

    violations = contract.validate_page(_mutated(mutate), CONTRACT)
    assert len(violations) == 1
    assert "non-nullable" in violations[0]


def test_null_arrival_time_is_not_a_violation():
    # Real, expected case (origin-stop row) -- must NOT be flagged.
    def mutate(payload):
        payload["data"][0]["attributes"]["arrival_time"] = None

    assert contract.validate_page(_mutated(mutate), CONTRACT) == []


def test_null_vehicle_relationship_is_not_a_violation():
    def mutate(payload):
        payload["data"][0]["relationships"]["vehicle"] = {"data": None}

    assert contract.validate_page(_mutated(mutate), CONTRACT) == []


def test_missing_required_relationship_key_is_a_violation():
    def mutate(payload):
        del payload["data"][0]["relationships"]["route"]

    violations = contract.validate_page(_mutated(mutate), CONTRACT)
    assert len(violations) == 1
    assert "relationships.route" in violations[0]
    assert "missing" in violations[0]


def test_relationship_id_not_a_string_is_a_violation():
    def mutate(payload):
        payload["data"][0]["relationships"]["route"]["data"]["id"] = 23  # int, not str

    violations = contract.validate_page(_mutated(mutate), CONTRACT)
    assert len(violations) == 1
    assert "relationships.route.data.id" in violations[0]


def test_wrong_resource_type_is_a_violation():
    def mutate(payload):
        payload["data"][0]["type"] = "prediction-v2"

    violations = contract.validate_page(_mutated(mutate), CONTRACT)
    assert len(violations) == 1
    assert "data[0].type" in violations[0]


def test_missing_required_top_level_field_is_a_violation():
    def mutate(payload):
        del payload["data"][0]["attributes"]

    violations = contract.validate_page(_mutated(mutate), CONTRACT)
    # Missing `attributes` itself, plus every attribute-level required-field
    # check treats the (now absent) dict as empty -- direction_id is the
    # only required attribute, so exactly two violations.
    assert any("missing required top-level field 'attributes'" in v for v in violations)
    assert any("attributes.direction_id" in v and "missing" in v for v in violations)


def test_missing_schedule_attribute_dict_is_a_violation():
    def mutate(payload):
        del payload["included"][0]["attributes"]

    violations = contract.validate_page(_mutated(mutate), CONTRACT)
    assert any("included[0](schedule)" in v and "attributes" in v for v in violations)


def test_null_schedule_arrival_time_is_not_a_violation():
    # Real, expected case (schedule resource for an origin stop).
    def mutate(payload):
        payload["included"][0]["attributes"]["arrival_time"] = None

    assert contract.validate_page(_mutated(mutate), CONTRACT) == []


def test_non_schedule_included_resources_are_not_validated():
    # map_rows never reads route/stop/trip/vehicle attributes out of
    # `included` -- only relationship ids elsewhere in `data` -- so a
    # malformed one of those resource types isn't a contract violation.
    def mutate(payload):
        payload["included"].append({"id": "route-1", "type": "route"})  # no attributes at all

    assert contract.validate_page(_mutated(mutate), CONTRACT) == []


def test_multiple_violations_across_predictions_are_all_reported():
    def mutate(payload):
        payload["data"][0]["attributes"]["direction_id"] = None
        payload["data"][1]["attributes"]["direction_id"] = 5

    violations = contract.validate_page(_mutated(mutate), CONTRACT)
    assert len(violations) == 2
    assert "data[0]" in violations[0]
    assert "data[1]" in violations[1]
