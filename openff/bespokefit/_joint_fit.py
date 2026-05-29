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

import os
import shutil
import socket
import subprocess
from contextlib import contextmanager
from typing import List, Optional, Tuple


def _has_qc_data(reference_data) -> bool:
    """A target is ready for a (local) joint fit when its reference data holds concrete
    QC records (``LocalQCData``) rather than the ``BespokeQCData`` placeholder. Duck-typed
    so this module needn't import the heavy schema classes."""
    return bool(getattr(reference_data, "qc_records", None))


def pool_parameters_and_targets(stages) -> Tuple[list, list]:
    """Pool the parameters and targets of several optimization stages.

    Parameters are de-duplicated by ``(type, smirks)`` with their fitted ``attributes``
    unioned, so a torsion shared by several molecules is fit once against all of their
    data. Targets are concatenated; each carries its own molecule's QC reference data.

    Raises:
        ValueError: if a target is missing concrete QC reference data.
    """
    parameters_by_key = {}
    ordered_keys = []
    targets = []

    for stage in stages:
        for parameter in stage.parameters:
            key = (parameter.type, parameter.smirks)
            if key not in parameters_by_key:
                parameters_by_key[key] = [parameter, set(parameter.attributes)]
                ordered_keys.append(key)
            else:
                parameters_by_key[key][1].update(parameter.attributes)

        for target in stage.targets:
            if not _has_qc_data(target.reference_data):
                raise ValueError(
                    "A fitting target is missing its QC reference data; the joint fit "
                    "needs the generated torsion drives. Ensure every optimization in "
                    "the series completed successfully."
                )
            targets.append(target)

    parameters = [
        parameter.copy(update={"attributes": attributes})
        for key in ordered_keys
        for parameter, attributes in [parameters_by_key[key]]
    ]

    return parameters, targets


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
