import logging
import math
from typing import Any, Dict, List

# Fix for OpenEye segfaults in forked processes
try:
    from openeye import oechem
    if oechem.OEChemIsLicensed() and oechem.OEGetMemPoolMode() == oechem.OEMemPoolMode_Default:
        oechem.OESetMemPoolMode(oechem.OEMemPoolMode_Mutexed | oechem.OEMemPoolMode_UnboundedCache)
except (ImportError, ModuleNotFoundError):
    pass

import psutil
import qcelemental
import qcengine
from celery.utils.log import get_task_logger
from openff.toolkit.topology import Atom, Molecule
from qcelemental.models import AtomicInput, AtomicResult
from qcelemental.models.common_models import DriverEnum
from qcelemental.models.procedures import (
    OptimizationInput,
    OptimizationResult,
    OptimizationSpecification,
    QCInputSpecification,
    TDKeywords,
    TorsionDriveInput,
    TorsionDriveResult,
)
from qcelemental.util import serialize
from qcengine.config import get_global

from openff.bespokefit.executor.services import current_settings
from openff.bespokefit.executor.utilities.celery import configure_celery_app
from openff.bespokefit.executor.utilities.redis import connect_to_default_redis
from openff.bespokefit.schema.tasks import OptimizationTask, Torsion1DTask

celery_app = configure_celery_app(
    "qcgenerator", connect_to_default_redis(validate=False)
)

_task_logger: logging.Logger = get_task_logger(__name__)


def _task_config() -> Dict[str, Any]:
    worker_settings = current_settings().qc_compute_settings

    n_cores = (
        get_global("ncores") if not worker_settings.n_cores else worker_settings.n_cores
    )
    max_memory = (
        (psutil.virtual_memory().total / (1024**3))
        if not worker_settings.max_memory
        else (
            worker_settings.max_memory
            * qcelemental.constants.conversion_factor("gigabyte", "gibibyte")
            * n_cores
        )
    )

    return dict(ncores=n_cores, nnodes=1, memory=round(max_memory, 3), retries=3)


def _select_atom(atoms: List[Atom]) -> int:
    """
    For a list of atoms chose the heaviest atom.
    """
    candidate = atoms[0]
    for atom in atoms:
        if atom.atomic_number > candidate.atomic_number:
            candidate = atom
    return candidate.molecule_atom_index


def _get_program_keywords(program: str) -> Dict[str, str]:
    """
    Generate a set of pre-defined keywords for the calculation based on the program.
    These are based on experience.

    Args:
        program: The name of the program which will run the calculation

    Returns:
         A dictionary of program specific keywords
    """
    keywords = {}
    if program.lower() == "xtb":
        # <https://github.com/openforcefield/openff-bespokefit/issues/238>
        keywords["verbosity"] = "muted"
    return keywords


# A molecule's torsion profile is dropped only if fewer than this fraction of its grid
# points produce a converged single-point energy; otherwise the failed points are simply
# dropped from the profile.
_MIN_SINGLE_POINT_CONVERGED_FRACTION = 0.5


def _single_point_torsion_energies(
    task: Torsion1DTask, td_result: TorsionDriveResult
) -> TorsionDriveResult:
    """Re-evaluate torsion drive grid energies at a higher level of theory.

    The geometries optimized during the drive (e.g. with xTB) are retained while each
    grid energy is replaced by a single-point energy computed with
    ``task.single_point_spec`` (e.g. DFT).

    Single points are computed per grid point and are *fault tolerant*: a grid point whose
    single point fails to converge is dropped (from both the energies and the geometries)
    with a warning, rather than failing the whole drive. The molecule is only failed if
    fewer than ``_MIN_SINGLE_POINT_CONVERGED_FRACTION`` of its grid points converge. The
    number of dropped points is recorded in the result's ``extras`` so the caller can
    report a summary.
    """
    spec = task.single_point_spec
    n_points = len(td_result.final_molecules)

    _task_logger.info(
        f"computing single-point energies at {spec.program}/{spec.model.method} "
        f"for {n_points} grid points"
    )

    final_energies = {}
    final_molecules = {}
    failed_grid_ids = []

    for grid_id, qc_molecule in td_result.final_molecules.items():
        single_point = qcengine.compute(
            AtomicInput(
                molecule=qc_molecule,
                driver=DriverEnum.energy,
                model=spec.model,
                keywords=_get_program_keywords(spec.program),
            ),
            spec.program,
            raise_error=False,
            task_config=_task_config(),
        )

        if getattr(single_point, "success", False):
            final_energies[grid_id] = float(single_point.return_result)
            final_molecules[grid_id] = qc_molecule
        else:
            failed_grid_ids.append(grid_id)
            error = getattr(single_point, "error", None)
            _task_logger.warning(
                f"single-point energy failed at grid point {grid_id}: "
                f"{getattr(error, 'error_message', error)}"
            )

    n_failed = len(failed_grid_ids)
    n_converged = len(final_energies)

    if n_failed:
        _task_logger.warning(
            f"{n_failed}/{n_points} single-point energies failed to converge at "
            f"{spec.program}/{spec.model.method} and were dropped"
        )

    if n_converged < max(2, math.ceil(_MIN_SINGLE_POINT_CONVERGED_FRACTION * n_points)):
        raise RuntimeError(
            f"only {n_converged}/{n_points} single-point energies converged at "
            f"{spec.program}/{spec.model.method} (below 50%); too few to build a torsion "
            f"profile for this molecule."
        )

    extras = dict(td_result.extras or {})
    extras["bespokefit_single_point_failures"] = {
        "n_failed": n_failed,
        "n_total": n_points,
        "failed_grid_ids": [str(grid_id) for grid_id in failed_grid_ids],
    }

    return TorsionDriveResult(
        **td_result.dict(exclude={"final_energies", "final_molecules", "extras"}),
        extras=extras,
        final_energies=final_energies,
        final_molecules=final_molecules,
    )


@celery_app.task(acks_late=True)
def compute_torsion_drive(task_json: str) -> TorsionDriveResult:
    """Runs a torsion drive using QCEngine."""

    task = Torsion1DTask.parse_raw(task_json)

    _task_logger.info(f"running 1D scan with {_task_config()}")

    molecule: Molecule = Molecule.from_smiles(task.smiles)
    molecule.generate_conformers(n_conformers=task.n_conformers)

    map_to_atom_index = {
        map_index: atom_index
        for atom_index, map_index in molecule.properties["atom_map"].items()
    }

    index_2 = map_to_atom_index[task.central_bond[0]]
    index_3 = map_to_atom_index[task.central_bond[1]]

    index_1_atoms = [
        atom
        for atom in molecule.atoms[index_2].bonded_atoms
        if atom.molecule_atom_index != index_3
    ]
    index_4_atoms = [
        atom
        for atom in molecule.atoms[index_3].bonded_atoms
        if atom.molecule_atom_index != index_2
    ]

    del molecule.properties["atom_map"]

    input_schema = TorsionDriveInput(
        keywords=TDKeywords(
            dihedrals=[
                (
                    _select_atom(index_1_atoms),
                    index_2,
                    index_3,
                    _select_atom(index_4_atoms),
                )
            ],
            grid_spacing=[task.grid_spacing],
            dihedral_ranges=[task.scan_range] if task.scan_range is not None else None,
        ),
        extras={
            "canonical_isomeric_explicit_hydrogen_mapped_smiles": molecule.to_smiles(
                isomeric=True, explicit_hydrogens=True, mapped=True
            )
        },
        initial_molecule=[
            molecule.to_qcschema(conformer=i) for i in range(molecule.n_conformers)
        ],
        input_specification=QCInputSpecification(
            model=task.model,
            driver=DriverEnum.gradient,
            keywords=_get_program_keywords(task.program),
        ),
        optimization_spec=OptimizationSpecification(
            procedure=task.optimization_spec.program,
            keywords={
                **task.optimization_spec.dict(exclude={"program", "constraints"}),
                "program": task.program,
            },
        ),
    )

    # run all torsiondrives through our custom procedure which handles parallel optimisations
    return_value = qcengine.compute_procedure(
        input_schema,
        "TorsionDriveParallel",
        raise_error=True,
        task_config=_task_config(),
    )

    if isinstance(return_value, TorsionDriveResult):
        _task_logger.info(
            f"1D TorsionDrive successfully completed in {return_value.provenance.wall_time}"
        )
        return_value = TorsionDriveResult(
            **return_value.dict(exclude={"optimization_history", "stdout", "stderr"}),
            optimization_history={},
        )

        if task.single_point_spec is not None:
            return_value = _single_point_torsion_energies(task, return_value)

    # noinspection PyTypeChecker
    return return_value.json()


@celery_app.task
def compute_optimization(
    task_json: str,
) -> List[OptimizationResult]:
    """Runs a set of geometry optimizations using QCEngine."""
    # TODO: should we only return the lowest energy optimization?
    # or the first optimisation to work?

    task = OptimizationTask.parse_raw(task_json)

    _task_logger.info(f"running opt with {_task_config()}")

    molecule: Molecule = Molecule.from_smiles(task.smiles)
    molecule.generate_conformers(n_conformers=task.n_conformers)

    input_schemas = [
        OptimizationInput(
            keywords={
                **task.optimization_spec.dict(exclude={"program", "constraints"}),
                "program": task.program,
            },
            extras={
                "canonical_isomeric_explicit_hydrogen_mapped_smiles": molecule.to_smiles(
                    isomeric=True, explicit_hydrogens=True, mapped=True
                )
            },
            input_specification=QCInputSpecification(
                model=task.model,
                driver=DriverEnum.gradient,
                keywords=_get_program_keywords(task.program),
            ),
            initial_molecule=molecule.to_qcschema(conformer=i),
        )
        for i in range(molecule.n_conformers)
    ]

    return_values = []

    for input_schema in input_schemas:
        return_value = qcengine.compute_procedure(
            input_schema,
            task.optimization_spec.program,
            raise_error=True,
            task_config=_task_config(),
        )

        if isinstance(return_value, OptimizationResult):
            # Strip the extra **heavy** data
            return_value = OptimizationResult(
                **return_value.dict(exclude={"trajectory", "stdout", "stderr"}),
                trajectory=[],
            )

        return_values.append(return_value)

    # noinspection PyTypeChecker
    return serialize(return_values, "json")


@celery_app.task
def compute_hessian(task_json: str) -> AtomicResult:
    """Runs a set of hessian evaluations using QCEngine."""
    raise NotImplementedError()
