from qcelemental.models.common_models import Model

from openff.bespokefit.schema.tasks import (
    SinglePointSpec,
    Torsion1DTask,
    Torsion1DTaskSpec,
)


def test_torsion_spec_single_point_default_none():
    spec = Torsion1DTaskSpec(program="xtb", model=Model(method="gfn2xtb", basis=None))
    assert spec.single_point_spec is None


def test_torsion_task_carries_single_point_spec():
    """The coordinator builds a ``Torsion1DTask`` via ``**spec.dict()``; ensure the
    single-point spec survives that construction and a JSON round-trip (the latter is
    used both for caching and for dispatching the task to the worker)."""
    spec = Torsion1DTaskSpec(
        program="xtb",
        model=Model(method="gfn2xtb", basis=None),
        single_point_spec=SinglePointSpec(
            program="psi4", model=Model(method="b3lyp-d3bj", basis="dzvp")
        ),
    )

    task = Torsion1DTask(
        smiles="[CH2:1]=[CH2:2]", central_bond=(1, 2), **spec.dict()
    )

    assert task.single_point_spec is not None
    assert task.single_point_spec.program == "psi4"
    assert task.single_point_spec.model.method == "b3lyp-d3bj"
    assert task.single_point_spec.model.basis == "dzvp"

    # The cache key and worker dispatch both serialize to JSON, so it must round-trip.
    assert (
        Torsion1DTask.parse_raw(task.json()).single_point_spec == task.single_point_spec
    )


def test_torsion_task_distinct_json_per_single_point_spec():
    """Tasks differing only in their single-point spec must serialize differently so
    they cache as separate QC calculations."""
    base = dict(
        smiles="[CH2:1]=[CH2:2]",
        central_bond=(1, 2),
        program="xtb",
        model=Model(method="gfn2xtb", basis=None),
    )

    xtb_only = Torsion1DTask(**base)
    with_dft = Torsion1DTask(
        **base,
        single_point_spec=SinglePointSpec(
            program="psi4", model=Model(method="b3lyp-d3bj", basis="dzvp")
        ),
    )

    assert xtb_only.json() != with_dft.json()
