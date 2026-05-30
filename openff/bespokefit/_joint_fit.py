"""Assemble and run a single (joint) ForceBalance optimization from the per-molecule
results of a whole series.

For a joint fit, every molecule's QC torsion drives are pooled into one optimization that
fits a single, *shared* set of (broad) torsion parameters -- so the common torsions of a
congeneric series get one consistent value fit against all of the data, rather than a
separate value per molecule. This is the multi-molecule fit that bespokefit does not do
out of the box: its executor fits one molecule per optimization, so here we re-use the
generated QC data and drive ``ForceBalanceOptimizer.optimize`` directly with a hand-built
stage whose targets span every molecule.

The pooling logic (:func:`pool_parameters_and_targets`) is duck-typed and standalone; the
schema construction and optimiser invocation import the heavier bespokefit/OpenFF objects
lazily so this module stays cheap to import.
"""

import logging
import os
import shutil
import socket
import subprocess
from contextlib import contextmanager
from typing import List, Optional, Tuple

_logger = logging.getLogger(__name__)


def _has_qc_data(reference_data) -> bool:
    """A target is ready for a (local) joint fit when its reference data holds concrete
    QC records (``LocalQCData``) rather than the ``BespokeQCData`` placeholder. Duck-typed
    so this module needn't import the heavy schema classes."""
    return bool(getattr(reference_data, "qc_records", None))


def _torsion_record_identity(record):
    """A stable identity for a torsion-drive QC record: ``(canonical fragment CMILES,
    scanned dihedral)``.

    A congeneric series shares scaffold torsions, so the same drive (after task
    canonicalization) is referenced by several molecules; this identity lets those be
    de-duplicated. Falls back to per-object identity (i.e. no de-dup) when the record
    can't be introspected, so it never collapses genuinely distinct drives.
    """
    initial = getattr(record, "initial_molecule", None)
    molecule = initial[0] if isinstance(initial, (list, tuple)) and initial else initial
    extras = getattr(molecule, "extras", None) or {}
    cmiles = extras.get("canonical_isomeric_explicit_hydrogen_mapped_smiles")

    keywords = getattr(record, "keywords", None)
    dihedrals = getattr(keywords, "dihedrals", None)
    dihedral = tuple(dihedrals[0]) if dihedrals else None

    if cmiles is None:
        _logger.debug(
            "torsion-drive record has no CMILES in its initial-molecule extras; falling "
            "back to object identity, so it will not be de-duplicated across the series."
        )
        return id(record)
    return (cmiles, dihedral)


def pool_parameters_and_targets(stages) -> Tuple[list, list]:
    """Pool the parameters and targets of several optimization stages.

    Parameters are de-duplicated by ``(type, smirks)`` with their fitted ``attributes``
    unioned, so a torsion shared by several molecules is fit once against all of their
    data. Torsion-drive QC records are **de-duplicated by identity** -- shared-scaffold
    torsions referenced by several molecules are kept once, so they are not over-weighted
    or evaluated repeatedly -- and all targets of a type are merged into a single target
    carrying the unique records. (ForceBalance names targets from a per-target record
    index, so keeping one target per molecule would produce colliding names like
    ``torsion-0`` from every molecule.)

    Raises:
        ValueError: if a target is missing concrete QC reference data.
    """
    parameters_by_key = {}
    ordered_keys = []
    seen_records = set()
    targets_by_type = {}
    type_order = []

    for stage in stages:
        for parameter in stage.parameters:
            key = (parameter.type, parameter.smirks)
            if key not in parameters_by_key:
                parameters_by_key[key] = [parameter, set(parameter.attributes)]
                ordered_keys.append(key)
            else:
                parameters_by_key[key][1].update(parameter.attributes)

        for target in stage.targets:
            reference_data = target.reference_data
            if not _has_qc_data(reference_data):
                raise ValueError(
                    "A fitting target is missing its QC reference data; the joint fit "
                    "needs the generated torsion drives. Ensure every optimization in "
                    "the series completed successfully."
                )

            unique_records = []
            for record in reference_data.qc_records:
                identity = _torsion_record_identity(record)
                if identity in seen_records:
                    continue
                seen_records.add(identity)
                unique_records.append(record)

            if not unique_records:
                continue

            target_type = getattr(target, "type", type(target).__name__)
            if target_type not in targets_by_type:
                # First target of this type becomes the template the records merge into.
                targets_by_type[target_type] = [target, []]
                type_order.append(target_type)
            targets_by_type[target_type][1].extend(unique_records)

    targets = [
        template.copy(
            update={
                "reference_data": template.reference_data.copy(
                    update={"qc_records": records}
                )
            }
        )
        for target_type in type_order
        for template, records in [targets_by_type[target_type]]
    ]

    parameters = [
        parameter.copy(update={"attributes": attributes})
        for key in ordered_keys
        for parameter, attributes in [parameters_by_key[key]]
    ]

    return parameters, targets


def count_single_point_failures(outputs) -> Tuple[int, int, int]:
    """Total the DFT single-point energies that failed to converge (and were dropped)
    across a series of completed bespoke results.

    Returns ``(n_failed, n_total, n_affected_drives)``, read from the per-drive failure
    record the QC worker stamps into each ``TorsionDriveResult``'s ``extras``. Drives
    shared across the series (same identity) are counted once -- matching the QC cache --
    so the totals reflect unique single-point computations. Works for both per-molecule
    and joint runs, and returns zeros when no single-point spec was used.
    """
    n_failed = n_total = n_affected = 0
    seen = set()

    for output in outputs:
        results = getattr(output, "results", None)
        schema = getattr(results, "input_schema", None) if results is not None else None
        if schema is None:
            continue

        for stage in schema.stages:
            for target in stage.targets:
                reference_data = getattr(target, "reference_data", None)
                for record in getattr(reference_data, "qc_records", None) or []:
                    identity = _torsion_record_identity(record)
                    if identity in seen:
                        continue
                    seen.add(identity)

                    info = (getattr(record, "extras", None) or {}).get(
                        "bespokefit_single_point_failures"
                    )
                    if not info:
                        continue
                    n_total += info.get("n_total", 0)
                    failed = info.get("n_failed", 0)
                    n_failed += failed
                    if failed:
                        n_affected += 1

    return n_failed, n_total, n_affected


def count_unique_torsions(outputs) -> Tuple[int, int]:
    """Across completed bespoke results, return ``(n_total, n_unique)`` torsion drives.

    A congeneric series shares scaffold torsions, which bespokefit's QC cache computes
    once and references from every molecule that contains them; ``n_total - n_unique`` is
    therefore the number of torsion drives reused from cache. Works in both per-molecule
    and joint modes; returns zeros when no torsions are present.
    """
    total = 0
    seen = set()

    for output in outputs:
        results = getattr(output, "results", None)
        schema = getattr(results, "input_schema", None) if results is not None else None
        if schema is None:
            continue

        for stage in schema.stages:
            for target in stage.targets:
                reference_data = getattr(target, "reference_data", None)
                for record in getattr(reference_data, "qc_records", None) or []:
                    total += 1
                    seen.add(_torsion_record_identity(record))

    return total, len(seen)


def build_joint_stage(per_molecule_schemas: List):
    """Build a single ``OptimizationStageSchema`` that fits shared parameters against the
    pooled targets of every per-molecule schema (using each schema's final stage).

    The optimizer and parameter hyperparameters are taken from the first schema's final
    stage, so the joint fit uses the same optimizer settings as the per-molecule fits.
    """
    from openff.bespokefit.schema.fitting import OptimizationStageSchema

    if not per_molecule_schemas:
        raise ValueError("No per-molecule schemas were provided for the joint fit.")

    stages = [schema.stages[-1] for schema in per_molecule_schemas]
    parameters, targets = pool_parameters_and_targets(stages)

    if not parameters:
        raise ValueError("No parameters to fit were found across the series.")
    if not targets:
        raise ValueError("No fitting targets with QC data were found across the series.")

    template = stages[0]

    return OptimizationStageSchema(
        optimizer=template.optimizer,
        parameters=parameters,
        parameter_hyperparameters=template.parameter_hyperparameters,
        targets=targets,
    )


def _find_free_port() -> int:
    """Return a currently free TCP port on the local host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def enable_work_queue(stage, port: int) -> None:
    """Configure ``stage``'s ForceBalance optimizer to distribute its target evaluations
    to Work Queue workers connecting on ``port``.

    ForceBalance turns on its Work Queue manager when ``wq_port`` is set, and bespokefit
    writes the optimizer ``extras`` verbatim into ``optimize.in`` -- so this simply adds
    the relevant options there.
    """
    extras = dict(stage.optimizer.extras)
    extras["wq_port"] = str(port)
    extras.setdefault("asynchronous", "")
    stage.optimizer = stage.optimizer.copy(update={"extras": extras})


@contextmanager
def work_queue_workers(n_workers: int, port: int, log_directory: str = "."):
    """Launch ``n_workers`` cctools ``work_queue_worker`` processes pointed at
    ``localhost:port`` (one core each) and terminate them on exit.

    Raises:
        RuntimeError: if the ``work_queue_worker`` executable (cctools) is not on PATH.
    """
    if shutil.which("work_queue_worker") is None:
        raise RuntimeError(
            "`work_queue_worker` was not found on the PATH. Parallel ForceBalance needs "
            "cctools / Work Queue; install it with `conda install -c conda-forge "
            "ndcctools` (or `mamba install ...`)."
        )

    processes = []
    logs = []
    try:
        for i in range(n_workers):
            log = open(os.path.join(log_directory, f"wq-worker-{i}.log"), "w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    ["work_queue_worker", "--cores=1", "localhost", str(port)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        yield
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
        for log in logs:
            log.close()


def _optimize_to_force_field(stage, initial_force_field, root_directory: str):
    from openff.toolkit.typing.engines.smirnoff import ForceField

    from openff.bespokefit.optimizers import get_optimizer

    optimizer = get_optimizer(stage.optimizer.type)
    results = optimizer.optimize(
        schema=stage,
        initial_force_field=initial_force_field,
        root_directory=root_directory,
    )

    if results.status != "success" or results.refit_force_field is None:
        error = results.error
        message = error.message if error is not None else "unknown error"
        raise RuntimeError(f"The joint optimization failed: {message}")

    return ForceField(results.refit_force_field, allow_cosmetic_attributes=True)


def run_joint_optimization(
    stage,
    initial_force_field,
    root_directory: str,
    n_workers: int = 0,
    wq_port: Optional[int] = None,
):
    """Run a single ForceBalance optimization for ``stage`` and return the refit
    ``ForceField``.

    If ``n_workers`` > 0 the fit is parallelized via ForceBalance's Work Queue: that many
    ``work_queue_worker`` processes are launched (on ``wq_port``, or an automatically
    chosen free port) to evaluate fitting targets concurrently, and torn down afterwards.

    Raises:
        RuntimeError: if the optimization does not complete successfully, or if
            ``n_workers`` > 0 but cctools is not installed.
    """
    if n_workers and n_workers > 0:
        port = wq_port or _find_free_port()
        enable_work_queue(stage, port)
        with work_queue_workers(n_workers, port):
            return _optimize_to_force_field(stage, initial_force_field, root_directory)

    return _optimize_to_force_field(stage, initial_force_field, root_directory)
