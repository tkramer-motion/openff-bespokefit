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


def drive_frequencies(outputs) -> dict:
    """Return ``{drive identity: number of molecule references}`` across the series, using
    the same identity as de-duplication. A drive referenced by >= 2 molecules is a shared
    (scaffold) torsion; a drive referenced once is unique (R-group)."""
    frequencies = {}

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
                    frequencies[identity] = frequencies.get(identity, 0) + 1

    return frequencies


def _torsion_profile_rmse(force_field, record) -> Optional[float]:
    """RMSE (kcal/mol) between the MM and QM *relative* torsion-energy profiles, with the
    MM energies evaluated as single points of ``force_field`` at the QM grid geometries.

    Returns ``None`` if the profile can't be evaluated, so the diagnostic never crashes a
    run. Requires OpenMM + the OpenFF toolkit, imported lazily.
    """
    try:
        import numpy
        import openmm
        from openff.toolkit import Molecule

        molecule_extras = getattr(record.initial_molecule[0], "extras", None) or {}
        cmiles = molecule_extras.get(
            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
        )
        if cmiles is None:
            return None

        final_energies = record.final_energies
        final_molecules = record.final_molecules
        grids = [grid for grid in final_energies if grid in final_molecules]
        if len(grids) < 2:
            return None

        molecule = Molecule.from_mapped_smiles(cmiles, allow_undefined_stereo=True)
        system = force_field.create_openmm_system(molecule.to_topology())
        context = openmm.Context(
            system,
            openmm.VerletIntegrator(1.0 * openmm.unit.femtoseconds),
            openmm.Platform.getPlatformByName("Reference"),
        )

        _BOHR_TO_NM = 0.052917721067
        _HARTREE_TO_KCAL = 627.5094740631

        qm_energies = []
        mm_energies = []
        for grid in grids:
            geometry = numpy.asarray(
                final_molecules[grid].geometry, dtype=float
            ).reshape(-1, 3)
            context.setPositions((geometry * _BOHR_TO_NM) * openmm.unit.nanometer)
            mm_energies.append(
                context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(openmm.unit.kilocalorie_per_mole)
            )
            qm_energies.append(final_energies[grid] * _HARTREE_TO_KCAL)

        qm = numpy.array(qm_energies)
        mm = numpy.array(mm_energies)
        qm -= qm.min()
        mm -= mm.min()
        return float(numpy.sqrt(numpy.mean((mm - qm) ** 2)))
    except Exception:
        return None


def summarize_fit_residuals(per_drive) -> dict:
    """Aggregate ``[(is_recurring, rmse_or_None), ...]`` into per-class RMSE statistics.

    Returns ``{"recurring": {...}, "unique": {...}}`` where each block has ``n`` (drives
    with an evaluable RMSE), ``mean``, ``median`` and ``max`` (kcal/mol; ``None`` when no
    drive in the class could be evaluated).
    """
    import statistics

    def _stats(values):
        values = [value for value in values if value is not None]
        if not values:
            return {"n": 0, "mean": None, "median": None, "max": None}
        return {
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    return {
        "recurring": _stats([rmse for recurring, rmse in per_drive if recurring]),
        "unique": _stats([rmse for recurring, rmse in per_drive if not recurring]),
    }


def torsion_fit_residual_report(
    force_field, targets, frequencies, recurring_threshold: int = 2
) -> dict:
    """Score how well ``force_field`` reproduces each unique torsion drive in ``targets``
    (MM-vs-QM profile RMSE), split into scaffold (recurring) and R-group (unique) drives.

    Use this after a joint fit to see whether the shared-SMIRKS fit reproduces the unique
    R-group torsions as well as the shared scaffold ones; if the unique RMSEs are much
    larger, a bespoke R-group treatment (the hybrid scheme) would likely help.
    """
    per_drive = []
    for target in targets:
        reference_data = getattr(target, "reference_data", None)
        for record in getattr(reference_data, "qc_records", None) or []:
            identity = _torsion_record_identity(record)
            is_recurring = frequencies.get(identity, 1) >= recurring_threshold
            per_drive.append((is_recurring, _torsion_profile_rmse(force_field, record)))

    return summarize_fit_residuals(per_drive)


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


def build_hybrid_stage(per_molecule_schemas: List) -> Tuple[object, str]:
    """Build one pooled ``OptimizationStageSchema`` for a **hybrid** fit and the initial
    force field it must be optimized against.

    The series is run with *bespoke* SMIRKS, so every scanned torsion has a molecule-
    specific parameter. Each **unique fragment torsion** (by identity) contributes one QC
    target, taken from the first molecule that has it, so:

    * a torsion shared across the series (same fragment) appears **once** -> fit once,
      consistently (the scaffold-joint half); and
    * a torsion unique to one molecule keeps its **own** bespoke parameter (the R-group-
      bespoke half).

    The subtlety that makes this correct: ChemPer mints a *different* bespoke SMIRKS per
    parent for the same fragment, and a congeneric series shares scaffold, so when every
    molecule's bespoke SMIRKS live in one force field OpenFF's "last match wins" rule can
    assign a scanned torsion a parameter that was minted from a *different* molecule.
    Marking the parameter we *think* owns a torsion (by SMIRKS-matching the central
    dihedral) then marks one that is never the applied one for ~half the torsions: it
    receives no gradient and ForceBalance leaves it at its base value, while the genuinely
    applied parameter is left unfit. So instead we build the complete merged force field
    first and then, for each scanned torsion, optimize **exactly the parameter the merged
    force field actually assigns to it** (via :meth:`label_molecules`). Parameters never
    assigned to any scanned torsion are pruned so unfit, base-valued duplicates cannot
    leak into the output.

    Returns ``(stage, initial_force_field_xml)`` where the force field is the base plus
    the bespoke parameters that are actually applied (with their initial values), so
    ForceBalance can look up each pooled parameter.
    """
    from openff.toolkit import ForceField, Molecule

    from openff.bespokefit.schema.fitting import OptimizationStageSchema
    from openff.bespokefit.schema.smirnoff import ProperTorsionSMIRKS

    if not per_molecule_schemas:
        raise ValueError("No per-molecule schemas were provided for the hybrid fit.")

    template = per_molecule_schemas[0].stages[-1]

    base_smirks = {
        parameter.smirks
        for parameter in ForceField(
            per_molecule_schemas[0].initial_force_field, allow_cosmetic_attributes=True
        )
        .get_parameter_handler("ProperTorsions")
        .parameters
    }

    # --- Phase 1: assemble the COMPLETE merged force field (base + every molecule's
    # bespoke torsions) and the de-duplicated set of scanned-torsion sites. -------------
    merged_force_field = ForceField(
        per_molecule_schemas[0].initial_force_field, allow_cosmetic_attributes=True
    )
    merged_handler = merged_force_field.get_parameter_handler("ProperTorsions")
    merged_smirks = {parameter.smirks for parameter in merged_handler.parameters}

    # The fit-spec objects (which carry *which* attributes to optimize), keyed by SMIRKS
    # across the whole series.
    fit_spec_by_smirks = {}
    for schema in per_molecule_schemas:
        for parameter in schema.stages[-1].parameters:
            fit_spec_by_smirks.setdefault(parameter.smirks, parameter)

    seen = set()
    targets_by_type = {}
    type_order = []
    scanned_sites = []  # (cmiles, central dihedral) per unique scanned torsion

    for schema in per_molecule_schemas:
        stage = schema.stages[-1]
        molecule_handler = ForceField(
            schema.initial_force_field, allow_cosmetic_attributes=True
        ).get_parameter_handler("ProperTorsions")
        # Every molecule's bespoke torsions must be present before we label, so the
        # assignment we read reflects the force field the fit will actually use.
        for parameter in molecule_handler.parameters:
            if parameter.smirks not in merged_smirks:
                merged_handler.add_parameter(parameter=parameter)
                merged_smirks.add(parameter.smirks)

        for target in stage.targets:
            if not _has_qc_data(target.reference_data):
                raise ValueError(
                    "A hybrid fitting target is missing its QC reference data."
                )

            new_records = []
            for record in target.reference_data.qc_records:
                identity = _torsion_record_identity(record)
                if identity in seen:
                    continue
                seen.add(identity)
                new_records.append(record)

                cmiles = (
                    getattr(record.initial_molecule[0], "extras", None) or {}
                ).get("canonical_isomeric_explicit_hydrogen_mapped_smiles")
                dihedrals = getattr(
                    getattr(record, "keywords", None), "dihedrals", None
                )
                if cmiles is not None and dihedrals:
                    scanned_sites.append((cmiles, tuple(dihedrals[0])))

            if new_records:
                # Merge every molecule's unique records into a single target per type.
                # ForceBalance names target sub-directories by a per-target record index
                # (torsion-0, torsion-1, ...), so keeping one target per molecule would
                # produce colliding names ("The list of target names is not unique!").
                target_type = getattr(target, "type", type(target).__name__)
                if target_type not in targets_by_type:
                    targets_by_type[target_type] = [target, []]
                    type_order.append(target_type)
                targets_by_type[target_type][1].extend(new_records)

    # --- Phase 2: optimize exactly the parameter the merged force field assigns to each
    # scanned torsion (see the docstring for why SMIRKS-matching is not enough). --------
    labels_by_cmiles = {}
    applied_smirks = []
    applied_set = set()
    unassigned = 0
    for cmiles, central in scanned_sites:
        labels = labels_by_cmiles.get(cmiles)
        if labels is None:
            molecule = Molecule.from_mapped_smiles(
                cmiles, allow_undefined_stereo=True
            )
            labels = merged_force_field.label_molecules(molecule.to_topology())[0][
                "ProperTorsions"
            ]
            labels_by_cmiles[cmiles] = labels

        # ValenceDict normalizes torsion keys (i,j,k,l)==(l,k,j,i) in __getitem__, but
        # not necessarily in .get(); index with a fallback so atom-ordering can't miss.
        try:
            parameter = labels[central]
        except KeyError:
            try:
                parameter = labels[tuple(reversed(central))]
            except KeyError:
                unassigned += 1
                continue
        if parameter.smirks not in applied_set:
            applied_set.add(parameter.smirks)
            applied_smirks.append(parameter.smirks)

    if unassigned:
        _logger.warning(
            "hybrid fit: %d scanned torsion(s) had no proper-torsion parameter assigned "
            "and were skipped.",
            unassigned,
        )

    # Parameters to optimize = exactly those assigned to a scanned torsion.
    pooled_parameters = [
        fit_spec_by_smirks.get(smirks)
        or ProperTorsionSMIRKS.from_smirnoff(merged_handler.parameters[smirks])
        for smirks in applied_smirks
    ]

    # Rebuild the initial force field as base + only the applied bespoke parameters, so
    # unfit (never-applied) duplicates do not leak into the output at their base values.
    # Dropping non-applied parameters cannot change which parameter wins for any scanned
    # torsion (the winner is kept), so the assignment above still holds.
    initial_force_field = ForceField(
        per_molecule_schemas[0].initial_force_field, allow_cosmetic_attributes=True
    )
    initial_handler = initial_force_field.get_parameter_handler("ProperTorsions")
    initial_smirks = {parameter.smirks for parameter in initial_handler.parameters}
    for smirks in applied_smirks:
        if smirks not in initial_smirks:
            initial_handler.add_parameter(parameter=merged_handler.parameters[smirks])
            initial_smirks.add(smirks)

    pooled_targets = [
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

    if not pooled_parameters:
        raise ValueError(
            "No bespoke parameters could be linked to the pooled targets for the hybrid "
            "fit."
        )
    if not pooled_targets:
        raise ValueError("No fitting targets with QC data were found across the series.")

    n_bespoke = sum(1 for smirks in applied_smirks if smirks not in base_smirks)
    _logger.info(
        "hybrid fit: optimizing %d parameter(s) (%d bespoke, %d inherited) actually "
        "assigned to %d scanned torsion(s).",
        len(pooled_parameters),
        n_bespoke,
        len(pooled_parameters) - n_bespoke,
        len(scanned_sites),
    )

    stage = OptimizationStageSchema(
        optimizer=template.optimizer,
        parameters=pooled_parameters,
        parameter_hyperparameters=template.parameter_hyperparameters,
        targets=pooled_targets,
    )
    return stage, initial_force_field.to_string(discard_cosmetic_attributes=True)


def stage_parameter_items(stage):
    """Yield ``(smirks, parameter)`` for a stage's parameters (duck-typed helper)."""
    for parameter in stage.parameters:
        yield parameter.smirks, parameter


def count_moved_parameters(initial_force_field, refit_force_field, stage) -> Tuple[int, int]:
    """Return ``(n_moved, n_total)`` proper-torsion parameters that the fit actually
    changed, comparing each optimized parameter's force constants before vs after.

    This is the automatic form of "did the fit move the parameters it was supposed to":
    a hybrid (or joint) fit in which the optimized parameters are not the ones the force
    field actually applies to the scanned torsions leaves them ungradiented at their
    initial values, so ``n_moved`` would be ~0. Returns ``(0, 0)`` if it cannot evaluate.
    """
    try:
        initial = initial_force_field.get_parameter_handler("ProperTorsions")
        refit = refit_force_field.get_parameter_handler("ProperTorsions")
    except Exception:
        return 0, 0

    def _ks(handler, smirks):
        try:
            parameter = handler.parameters[smirks]
        except Exception:
            return None
        return tuple(
            round(value.m_as("kilojoule / mole"), 6) for value in (parameter.k or [])
        )

    n_moved = n_total = 0
    for parameter in getattr(stage, "parameters", []):
        before = _ks(initial, parameter.smirks)
        after = _ks(refit, parameter.smirks)
        if before is None or after is None:
            continue
        n_total += 1
        if before != after:
            n_moved += 1

    return n_moved, n_total


def _find_free_port() -> int:
    """Return a currently free TCP port on the local host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def enable_work_queue(stage, port: int) -> None:
    """Configure ``stage`` to distribute its ForceBalance target evaluations to Work Queue
    workers connecting on ``port``.

    Two things are required, and ``stage`` carries both as ``extras`` that bespokefit
    writes verbatim into ``optimize.in``:

    * ``wq_port`` on the optimizer, which starts ForceBalance's Work Queue manager; and
    * ``remote`` on **every target**, which tells ForceBalance to farm that target's
      evaluations out to a worker. Without ``remote`` ForceBalance evaluates the targets
      locally (serially, on one CPU) while the workers sit idle -- so both are set here.
    """
    extras = dict(stage.optimizer.extras)
    extras["wq_port"] = str(port)
    extras.setdefault("asynchronous", "")
    stage.optimizer = stage.optimizer.copy(update={"extras": extras})

    remote_targets = []
    for target in stage.targets:
        target_extras = dict(getattr(target, "extras", None) or {})
        target_extras.setdefault("remote", "1")
        remote_targets.append(target.copy(update={"extras": target_extras}))
    stage.targets = remote_targets


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

    # Always start from a clean scratch directory. ForceBalance caches a remote force
    # field (``forcefield-remote``) and other state keyed to the parameter vector; reusing
    # a directory from a previous run with a different number of parameters (e.g. a joint
    # run, then a hybrid run) makes ForceBalance compare mismatched ``mvals`` and crash.
    shutil.rmtree(root_directory, ignore_errors=True)

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
