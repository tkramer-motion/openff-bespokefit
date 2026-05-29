"""Tests for the joint-fit pooling logic.

``pool_parameters_and_targets`` is duck-typed, so it is exercised here with lightweight
stubs (no OpenFF toolkit needed). The schema construction and ForceBalance invocation
(``build_joint_stage`` / ``run_joint_optimization``) require the full environment and are
covered by integration runs rather than unit tests.
"""

import pytest

from openff.bespokefit import _joint_fit
from openff.bespokefit._joint_fit import (
    _find_free_port,
    enable_work_queue,
    pool_parameters_and_targets,
    work_queue_workers,
)


class _Param:
    def __init__(self, type, smirks, attributes):
        self.type = type
        self.smirks = smirks
        self.attributes = set(attributes)

    def copy(self, update=None):
        new = _Param(self.type, self.smirks, set(self.attributes))
        for key, value in (update or {}).items():
            setattr(new, key, value)
        return new


class _RefData:
    def __init__(self, qc_records):
        self.qc_records = qc_records


class _Placeholder:
    """Stands in for BespokeQCData (no concrete QC records yet)."""

    def __init__(self):
        self.spec = object()


class _Target:
    def __init__(self, reference_data):
        self.reference_data = reference_data


class _Stage:
    def __init__(self, parameters, targets):
        self.parameters = parameters
        self.targets = targets


def _target():
    return _Target(_RefData(qc_records=["torsion-drive"]))


def test_pool_dedups_and_unions_attributes():
    """The same SMIRKS from two molecules collapses to one parameter whose attributes are
    the union, and all targets are pooled."""
    stage_a = _Stage(
        parameters=[_Param("ProperTorsions", "[*:1]-[*:2]-[*:3]-[*:4]", {"k1"})],
        targets=[_target()],
    )
    stage_b = _Stage(
        parameters=[_Param("ProperTorsions", "[*:1]-[*:2]-[*:3]-[*:4]", {"k2"})],
        targets=[_target()],
    )

    parameters, targets = pool_parameters_and_targets([stage_a, stage_b])

    assert len(parameters) == 1
    assert parameters[0].smirks == "[*:1]-[*:2]-[*:3]-[*:4]"
    assert parameters[0].attributes == {"k1", "k2"}
    assert len(targets) == 2


def test_pool_keeps_distinct_smirks_in_order():
    stage = _Stage(
        parameters=[
            _Param("ProperTorsions", "A", {"k1"}),
            _Param("ProperTorsions", "B", {"k1"}),
        ],
        targets=[_target()],
    )

    parameters, targets = pool_parameters_and_targets([stage])

    assert [p.smirks for p in parameters] == ["A", "B"]
    assert len(targets) == 1


def test_pool_distinguishes_by_type():
    stage = _Stage(
        parameters=[
            _Param("ProperTorsions", "S", {"k1"}),
            _Param("ImproperTorsions", "S", {"k1"}),
        ],
        targets=[_target()],
    )

    parameters, _ = pool_parameters_and_targets([stage])
    assert len(parameters) == 2


def test_pool_raises_without_qc_data():
    stage = _Stage(
        parameters=[_Param("ProperTorsions", "S", {"k1"})],
        targets=[_Target(_Placeholder())],
    )

    with pytest.raises(ValueError, match="missing its QC reference data"):
        pool_parameters_and_targets([stage])


def test_pool_raises_with_empty_qc_records():
    stage = _Stage(
        parameters=[_Param("ProperTorsions", "S", {"k1"})],
        targets=[_Target(_RefData(qc_records=[]))],
    )

    with pytest.raises(ValueError, match="missing its QC reference data"):
        pool_parameters_and_targets([stage])


# --- Work Queue helpers --------------------------------------------------------------


class _Optimizer:
    def __init__(self, extras=None):
        self.extras = dict(extras or {})

    def copy(self, update=None):
        new = _Optimizer(self.extras)
        for key, value in (update or {}).items():
            setattr(new, key, value)
        return new


class _StageWithOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer


def test_enable_work_queue_sets_extras():
    stage = _StageWithOptimizer(_Optimizer())
    enable_work_queue(stage, 9876)
    assert stage.optimizer.extras["wq_port"] == "9876"
    assert "asynchronous" in stage.optimizer.extras


def test_find_free_port_returns_usable_port():
    port = _find_free_port()
    assert isinstance(port, int) and port > 0


def test_work_queue_workers_requires_cctools(monkeypatch):
    monkeypatch.setattr(_joint_fit.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="cctools"):
        with work_queue_workers(2, 9999):
            pass
