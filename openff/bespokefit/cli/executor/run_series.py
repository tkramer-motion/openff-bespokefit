import time
from typing import List, Optional, Tuple

import click
import click.exceptions
import rich
from rich import pretty
from rich.padding import Padding

from openff.bespokefit.cli.executor.launch import (
    launch_options,
    validate_redis_connection,
)
from openff.bespokefit.cli.executor.submit import _submit, submit_options
from openff.bespokefit.cli.utilities import (
    create_command,
    exit_with_messages,
    print_header,
)


def _print_torsion_fit_diagnostic(console, report) -> None:
    """Print the scaffold (recurring) vs R-group (unique) MM-vs-QM RMSE summary, with a
    verdict on whether a bespoke R-group (hybrid) treatment would likely help."""
    recurring = report["recurring"]
    unique = report["unique"]

    if not recurring["n"] and not unique["n"]:
        return

    console.print(
        Padding(
            "torsion-fit diagnostic (MM vs QM relative energies, kcal/mol):",
            (1, 0, 0, 0),
        )
    )
    for label, stats in [
        ("scaffold (recurring)", recurring),
        ("R-group (unique)   ", unique),
    ]:
        if stats["n"]:
            console.print(
                f"    {label}: [blue]{stats['n']}[/blue] drives — mean "
                f"[blue]{stats['mean']:.2f}[/blue], median "
                f"[blue]{stats['median']:.2f}[/blue], max [blue]{stats['max']:.2f}[/blue]"
            )
        else:
            console.print(f"    {label}: none evaluated")

    if recurring["n"] and unique["n"] and recurring["mean"] and unique["mean"]:
        ratio = unique["mean"] / recurring["mean"] if recurring["mean"] > 0 else None
        if unique["mean"] > 0.5 and ratio and ratio > 1.5:
            console.print(
                f"    [yellow]→[/yellow] R-group torsions fit ~[blue]{ratio:.1f}x[/blue] "
                f"worse than the scaffold — a bespoke R-group (hybrid) treatment would "
                f"likely help."
            )
        else:
            console.print(
                "    [green]→[/green] R-group and scaffold torsions fit comparably; the "
                "joint fit looks adequate."
            )


def _run_series_cli(
    input_file_path: Optional[List[str]],
    molecule_smiles: Optional[List[str]],
    output_file_path: str,
    base_tmd_ff_path: str,
    openff_base: str,
    joint: bool,
    hybrid: bool,
    forcebalance_workers: int,
    forcebalance_wq_port: Optional[int],
    diagnose_fit: bool,
    force_field_path: Optional[str],
    target_torsion_smirks: Tuple[str],
    default_qc_spec: Optional[Tuple[str, str, str]],
    single_point_qc_spec: Optional[Tuple[str, str, str]],
    workflow_name: Optional[str],
    workflow_file_name: Optional[str],
    directory: Optional[str],
    n_fragmenter_workers: int,
    n_qc_compute_workers: int,
    qc_compute_n_cores: Optional[int],
    qc_compute_max_mem: Optional[float],
    n_optimizer_workers: int,
    launch_redis_if_unavailable: bool,
):
    """Fit torsions for a series of molecules and merge them onto a base Timemachine
    (tmd) force field for use in an FEP calculation.

    A temporary bespoke executor is launched and every molecule is submitted and fit.
    By default each molecule is fit independently (bespoke, molecule-specific torsions)
    and the results are merged. With ``--joint``, the series is fit *together*: the base
    force field's broad torsion SMIRKS are shared across all molecules and a single
    ForceBalance optimization is run over the pooled QC data, giving one consistent set
    of torsion parameters for the whole map (optionally parallelized with
    ``--forcebalance-workers``).

    Either way the resulting torsions are merged onto the base tmd force field given by
    ``--base-tmd-ff`` (whose charges and other parameters are preserved). Where a fit
    torsion matches an existing base torsion the newly fit parameters are used.
    """
    pretty.install()

    console = rich.get_console()
    print_header(console)

    from openff.toolkit import ForceField

    from openff.bespokefit._joint_fit import (
        count_single_point_failures,
        count_unique_torsions,
        drive_frequencies,
        torsion_fit_residual_report,
    )
    from openff.bespokefit._tmd import (
        collect_fitted_torsions,
        load_tmd_force_field,
        merge_torsions_into_tmd,
        save_tmd_force_field,
        tmd_proper_torsion_smirks,
    )
    from openff.bespokefit.executor import (
        BespokeExecutor,
        BespokeFitClient,
        BespokeWorkerConfig,
    )
    from openff.bespokefit.executor.services import current_settings
    from openff.bespokefit.executor.utilities import handle_common_errors

    # Fail fast on bad inputs *before* launching the executor, and warn if the initial
    # force field's torsions won't line up with the base tmd force field (which would
    # mean fit torsions get appended rather than overriding the base).
    try:
        base_tmd = load_tmd_force_field(base_tmd_ff_path)
    except (OSError, ValueError, SyntaxError) as e:
        exit_with_messages(
            f"[[red]ERROR[/red]] could not read the base tmd force field "
            f"[repr.filename]{base_tmd_ff_path}[/repr.filename]: {e}",
            console=console,
            exit_code=2,
        )

    try:
        fit_force_field = ForceField(force_field_path, allow_cosmetic_attributes=True)
    except Exception as e:
        exit_with_messages(
            f"[[red]ERROR[/red]] could not load the initial force field "
            f"[repr.filename]{force_field_path}[/repr.filename]: {e}",
            console=console,
            exit_code=2,
        )

    if forcebalance_workers and not (joint or hybrid):
        console.print(
            Padding(
                "[[yellow]![/yellow]] --forcebalance-workers only applies to --joint / "
                "--hybrid fits and will be ignored; per-molecule fits already run in "
                "parallel across --n-optimizer-workers.",
                (1, 0, 1, 0),
            )
        )

    # Fail now (not after hours of QC) if parallel ForceBalance was requested but cctools
    # is unavailable.
    if (joint or hybrid) and forcebalance_workers:
        import shutil

        if shutil.which("work_queue_worker") is None:
            exit_with_messages(
                "[[red]ERROR[/red]] --forcebalance-workers needs cctools / Work Queue, "
                "but `work_queue_worker` was not found on the PATH. Install it with "
                "`conda install -c conda-forge ndcctools`, or drop --forcebalance-workers "
                "to run ForceBalance serially.",
                console=console,
                exit_code=2,
            )

    fit_torsion_smirks = {
        parameter.smirks
        for parameter in fit_force_field.get_parameter_handler(
            "ProperTorsions"
        ).parameters
    }
    base_torsion_smirks = tmd_proper_torsion_smirks(base_tmd)
    if fit_torsion_smirks:
        overlap = len(fit_torsion_smirks & base_torsion_smirks) / len(
            fit_torsion_smirks
        )
        if overlap < 0.9:
            console.print(
                Padding(
                    f"[[yellow]![/yellow]] only [blue]{overlap:.0%}[/blue] of the "
                    f"initial force field's torsion SMIRKS "
                    f"([repr.filename]{force_field_path}[/repr.filename]) are present "
                    f"in the base tmd force field. Fit torsions whose SMIRKS are absent "
                    f"will be appended rather than overriding the base — if the base "
                    f"derives from a different OpenFF version, set --force-field to "
                    f"match its lineage.",
                    (1, 0, 1, 0),
                )
            )

    executor_status = console.status("launching the bespoke executor")
    executor_status.start()

    validate_redis_connection(console, allow_existing=False)
    settings = current_settings()
    client = BespokeFitClient(settings=settings)

    successful_outputs = []
    failures = []

    # The executor only needs to be alive while QC is generated and the per-molecule
    # workflows run; the (potentially long, parallel) joint fit happens afterwards, once
    # the executor's workers have been released.
    with BespokeExecutor(
        directory=directory,
        n_fragmenter_workers=n_fragmenter_workers,
        n_qc_compute_workers=n_qc_compute_workers,
        qc_compute_worker_config=BespokeWorkerConfig(
            n_cores="auto" if not qc_compute_n_cores else qc_compute_n_cores,
            max_memory="auto" if not qc_compute_max_mem else qc_compute_max_mem,
        ),
        n_optimizer_workers=n_optimizer_workers,
        launch_redis_if_unavailable=launch_redis_if_unavailable,
    ):
        executor_status.stop()
        console.print("[[green]✓[/green]] bespoke executor launched")
        console.line()

        with handle_common_errors(console) as error_state:
            response_ids = _submit(
                console=console,
                input_file_path=input_file_path,
                molecule_smiles=molecule_smiles,
                force_field_path=force_field_path,
                target_torsion_smirks=target_torsion_smirks,
                default_qc_spec=default_qc_spec,
                single_point_qc_spec=single_point_qc_spec,
                workflow_name=workflow_name,
                workflow_file_name=workflow_file_name,
                allow_multiple_molecules=True,
                save_submission=False,
                broad_smirks=joint and not hybrid,
                skip_optimization=joint or hybrid,
            )

            console.print(Padding("3. running the fitting pipeline", (1, 0, 1, 0)))

            for response_id in response_ids:
                result = client.wait_until_complete(
                    optimization_id=response_id, console=console
                )

                # The coordinator finalizes an optimization (attaches the fitted force
                # field and moves the stage running -> complete) in a cycle that can lag
                # behind the stage's status flipping to "success". If we read the result
                # inside that window we see status != "errored" but bespoke_force_field
                # is None. Re-poll briefly -- while the executor (and coordinator) is
                # still alive -- so a last-finishing molecule is not dropped by a
                # finalize/shutdown race.
                finalize_deadline = time.monotonic() + 60.0
                while (
                    result.status != "errored"
                    and result.bespoke_force_field is None
                    and time.monotonic() < finalize_deadline
                ):
                    time.sleep(2.0)
                    result = client.get_optimization(optimization_id=response_id)

                if result.status == "errored" or result.bespoke_force_field is None:
                    # Surface the actual stage state instead of a bare "unknown error"
                    # so a finalize race (optimization succeeded but no force field was
                    # persisted) is distinguishable from a genuine QC/chemistry failure.
                    if result.error:
                        reason = result.error
                    else:
                        stage_states = ", ".join(
                            f"{stage.type}={stage.status}" for stage in result.stages
                        )
                        reason = (
                            "no bespoke force field was returned even though no stage "
                            f"errored (stages: {stage_states}) -- this is a coordinator "
                            "finalize race, not a chemistry/QC failure"
                        )
                    failures.append((response_id, reason))
                    continue

                successful_outputs.append(result)

        if error_state["has_errored"]:
            raise click.exceptions.Exit(code=2)

    # --- the executor (and any redis it launched) is now shut down ---------------------

    for response_id, error in failures:
        console.print(
            Padding(
                f"[[yellow]![/yellow]] skipping [blue]{response_id}[/blue]: {error}",
                (0, 0, 0, 1),
            )
        )

    if not successful_outputs:
        exit_with_messages(
            "[[red]ERROR[/red]] none of the optimizations completed successfully",
            console=console,
            exit_code=2,
        )

    # Report how much the congeneric series shared (QC computed once, reused elsewhere).
    n_torsions_total, n_torsions_unique = count_unique_torsions(successful_outputs)
    if n_torsions_total:
        console.print(
            f"[[green]✓[/green]] computed [blue]{n_torsions_unique}[/blue] unique "
            f"torsion drives, reused [blue]{n_torsions_total - n_torsions_unique}[/blue] "
            f"from cache (across [blue]{len(successful_outputs)}[/blue] molecule(s))"
        )

    console.print(Padding("4. building the tmd force field", (1, 0, 1, 0)))

    if joint or hybrid:
        from openff.bespokefit._joint_fit import (
            build_hybrid_stage,
            build_joint_stage,
            count_moved_parameters,
            run_joint_optimization,
        )

        input_schemas = []
        for output in successful_outputs:
            results = output.results
            if results is None or results.input_schema is None:
                exit_with_messages(
                    "[[red]ERROR[/red]] a completed optimization did not return its "
                    "input schema with QC data, which a pooled fit requires.",
                    console=console,
                    exit_code=2,
                )
            input_schemas.append(results.input_schema)

        mode_name = "hybrid" if hybrid else "joint"
        worker_note = (
            f" using [blue]{forcebalance_workers}[/blue] ForceBalance worker(s)"
            if forcebalance_workers
            else ""
        )
        console.print(
            f"running a single {mode_name} ForceBalance fit over "
            f"[blue]{len(input_schemas)}[/blue] molecule(s){worker_note}"
        )

        if hybrid:
            # Bespoke SMIRKS pooled by fragment identity; each scanned torsion keeps its
            # own bespoke parameter, shared scaffold torsions appear once. The merged
            # force field (base + every contributing molecule's bespoke parameters) is
            # what ForceBalance optimizes; the fitted torsions are new vs the pure base.
            stage, initial_force_field_xml = build_hybrid_stage(input_schemas)
            pooled_initial_ff = ForceField(
                initial_force_field_xml, allow_cosmetic_attributes=True
            )
            base_openff = ForceField(force_field_path, allow_cosmetic_attributes=True)
        else:
            stage = build_joint_stage(input_schemas)
            pooled_initial_ff = ForceField(
                input_schemas[0].initial_force_field, allow_cosmetic_attributes=True
            )
            base_openff = pooled_initial_ff

        with console.status(f"running the {mode_name} ForceBalance optimization"):
            refit_force_fields = [
                run_joint_optimization(
                    stage,
                    pooled_initial_ff,
                    root_directory="joint-fit",
                    n_workers=forcebalance_workers,
                    wq_port=forcebalance_wq_port,
                )
            ]

        # Confirm the fit actually moved the parameters it targeted. If the optimized
        # parameters are not the ones the force field applies to the scanned torsions
        # they receive no gradient and stay at their initial values -- the failure mode
        # the hybrid assignment logic exists to prevent -- so surface it loudly.
        n_moved, n_target = count_moved_parameters(
            pooled_initial_ff, refit_force_fields[0], stage
        )
        if n_target:
            marker = "green]✓" if n_moved == n_target else "yellow]!"
            console.print(
                f"[[{marker}[/] the {mode_name} fit moved [blue]{n_moved}[/blue]/"
                f"[blue]{n_target}[/blue] targeted torsion parameter(s) off their "
                f"initial values"
            )
            if n_moved < n_target:
                console.print(
                    Padding(
                        f"[[yellow]![/yellow]] {n_target - n_moved} parameter(s) did not "
                        f"move -- they were not the parameter applied to their scanned "
                        f"torsion, so the fit could not constrain them",
                        (0, 0, 0, 1),
                    )
                )

        if diagnose_fit:
            with console.status(
                f"scoring the {mode_name} fit against QM (MM vs QM RMSE)"
            ):
                fit_report = torsion_fit_residual_report(
                    refit_force_fields[0],
                    stage.targets,
                    drive_frequencies(successful_outputs),
                )
            _print_torsion_fit_diagnostic(console, fit_report)
    else:
        # Compare against the force field the fits actually started from (so the
        # "what changed" detection stays consistent with --force-field), unless an
        # explicit --openff-base override was given.
        base_ff_name = openff_base
        if base_ff_name is None:
            first_results = successful_outputs[0].results
            base_ff_name = (
                first_results.input_schema.initial_force_field
                if first_results is not None and first_results.input_schema is not None
                else force_field_path
            )
        base_openff = ForceField(base_ff_name, allow_cosmetic_attributes=True)
        refit_force_fields = [
            output.bespoke_force_field for output in successful_outputs
        ]

    fitted_torsions = collect_fitted_torsions(refit_force_fields, base_openff)
    console.print(
        f"[[green]✓[/green]] collected [blue]{len(fitted_torsions)}[/blue] fit "
        f"torsions from [blue]{len(successful_outputs)}[/blue] molecule(s)"
    )

    combined, added, replaced = merge_torsions_into_tmd(base_tmd, fitted_torsions)
    save_tmd_force_field(combined, output_file_path)

    console.print(
        f"[[green]✓[/green]] appended [blue]{len(added)}[/blue] and overrode "
        f"[blue]{len(replaced)}[/blue] torsion(s) in the base tmd force field"
    )
    console.print(
        Padding(
            f"the tmd force field has been saved to "
            f"[repr.filename]{output_file_path}[/repr.filename]",
            (1, 0, 1, 0),
        )
    )

    # End-of-run summary of any DFT single points that failed to converge and were
    # dropped (only relevant when --single-point-qc-spec was used).
    n_sp_failed, n_sp_total, n_sp_drives = count_single_point_failures(successful_outputs)
    if n_sp_total:
        if n_sp_failed:
            console.print(
                f"[[yellow]![/yellow]] single-point summary: [blue]{n_sp_failed}[/blue]"
                f"/[blue]{n_sp_total}[/blue] DFT single points failed to converge and "
                f"were dropped, across [blue]{n_sp_drives}[/blue] torsion drive(s)"
            )
        else:
            console.print(
                f"[[green]✓[/green]] single-point summary: all [blue]{n_sp_total}[/blue]"
                f" DFT single points converged"
            )


__run_series_options = [
    *submit_options(
        allow_multiple_molecules=True,
        force_field_default="openff_unconstrained-2.0.0.offxml",
    )
]
__run_series_options.insert(
    4,
    click.option(
        "--output",
        "output_file_path",
        type=click.Path(exists=False, file_okay=True, dir_okay=False),
        help="The path [.py] to write the combined tmd force field to.",
        default="bespoke-tmd-forcefield.py",
        show_default=True,
    ),
)
__run_series_options.insert(
    5,
    click.option(
        "--base-tmd-ff",
        "base_tmd_ff_path",
        type=click.Path(exists=True, file_okay=True, dir_okay=False),
        help="The base tmd-format force field (a Python dict literal) to merge the "
        "fit torsions onto. Its charges and all other parameters are preserved.",
        required=True,
    ),
)
__run_series_options.insert(
    6,
    click.option(
        "--openff-base",
        "openff_base",
        type=click.STRING,
        default=None,
        help="Override for the force field used to tell newly fit torsions apart from "
        "inherited generic torsions. Defaults to whatever --force-field the fits started "
        "from. Ignored with --joint (the fit's initial force field is always used).",
    ),
)
__run_series_options.insert(
    7,
    click.option(
        "--joint/--per-molecule",
        "joint",
        default=False,
        show_default=True,
        help="Fit the whole series jointly: share the base force field's broad torsion "
        "SMIRKS across all molecules and run a single ForceBalance optimization over "
        "the pooled QC data (one consistent parameter set for the FEP map). The default "
        "(--per-molecule) fits each molecule independently and merges the results.",
    ),
)
__run_series_options.insert(
    8,
    click.option(
        "--hybrid/--no-hybrid",
        "hybrid",
        default=False,
        show_default=True,
        help="Hybrid fit (takes precedence over --joint): use bespoke SMIRKS but run one "
        "pooled ForceBalance fit in which each unique fragment torsion is fit once -- "
        "shared scaffold torsions consistently, R-group torsions with their own bespoke "
        "parameters. Best when R-group torsions fit poorly under --joint (see the "
        "torsion-fit diagnostic).",
    ),
)
__run_series_options.insert(
    9,
    click.option(
        "--forcebalance-workers",
        "forcebalance_workers",
        type=click.INT,
        default=0,
        show_default=True,
        help="Parallelize the joint ForceBalance fit by launching this many cctools "
        "`work_queue_worker` processes (one core each) to evaluate fitting targets "
        "concurrently. 0 runs ForceBalance serially. Requires cctools "
        "(`conda install -c conda-forge ndcctools`) and only applies with --joint.",
    ),
)
__run_series_options.insert(
    10,
    click.option(
        "--forcebalance-wq-port",
        "forcebalance_wq_port",
        type=click.INT,
        default=None,
        help="TCP port for the ForceBalance Work Queue manager. Defaults to an "
        "automatically chosen free port.",
    ),
)
__run_series_options.insert(
    11,
    click.option(
        "--diagnose-fit/--no-diagnose-fit",
        "diagnose_fit",
        default=True,
        show_default=True,
        help="After a --joint fit, score the fitted force field against the QM torsion "
        "drives (MM-vs-QM RMSE) split by scaffold (recurring) vs R-group (unique) "
        "torsions, to show whether a bespoke R-group treatment would help. Adds a short "
        "post-fit step (needs OpenMM).",
    ),
)
__run_series_options.extend(launch_options(directory=None))

run_series_cli = create_command(
    click_command=click.command("run-series"),
    click_options=__run_series_options,
    func=_run_series_cli,
)
