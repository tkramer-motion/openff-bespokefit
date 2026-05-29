# Joint torsion fitting for FEP (xTB drive → DFT single point → parallel ForceBalance)

This fork adds `openff-bespoke executor run-series`, a single command that fits torsions
for a **series** of molecules and writes one combined **Timemachine (tmd)** force-field
file you can drop straight into an FEP map.

It pulls together three things:

1. **xTB torsion drives with DFT single points** — each grid point is geometry-optimized
   cheaply with xTB, then re-evaluated with a single-point DFT energy (`--default-qc-spec`
   + `--single-point-qc-spec`).
2. **A joint fit** (`--joint`) — all molecules' torsion data are pooled into **one**
   ForceBalance optimization that fits the base force field's *shared, broad* torsion
   SMIRKS, giving one consistent parameter set across the whole map.
3. **Parallel ForceBalance** (`--forcebalance-workers`) — the joint fit's target
   evaluations are distributed across worker processes via cctools / Work Queue, because
   a single joint fit is otherwise the slow step.

The fit torsions are then merged onto your base tmd force field (its charges and all
other parameters are preserved).

---

## Prerequisites

- The bespokefit conda environment (this fork), with the QC engines you intend to use
  (`xtb`, `psi4`, …) installed.
- A **base tmd force field** as a Python dict literal (e.g. `project_base.py`) — this is the
  file the bespoke torsions are merged onto, and it supplies the charges + everything
  else for FEP.
- **cctools / Work Queue** — only needed for `--forcebalance-workers` (parallel
  ForceBalance). See below.

---

## What cctools / Work Queue is, and how to install it

**Work Queue** is a lightweight manager/worker framework from **cctools** (the
Cooperative Computing Tools, from Notre Dame). It lets one "manager" process hand small
tasks out to many "worker" processes.

**ForceBalance has built-in Work Queue support.** When ForceBalance is told to listen on
a Work Queue port (`wq_port`), it offloads its expensive per-target evaluations — the
energy/gradient computations it runs every optimizer iteration — to any
`work_queue_worker` processes that connect to that port. For a joint fit with many
targets and many parameters this is the difference between hours and minutes, and it is
the standard way the OpenFF "Sage" force fields are fit.

`run-series --joint --forcebalance-workers N` automates this: it sets `wq_port` on the
ForceBalance job and launches **N** `work_queue_worker` processes (one core each) on the
local machine, then tears them down when the fit finishes.

### Install

cctools must be installed **in the same conda environment as bespokefit/ForceBalance**
(the workers run the OpenMM/ForceBalance target evaluations, so they need that stack):

```bash
conda install -c conda-forge ndcctools
# or
mamba install -c conda-forge ndcctools
```

Verify it is on your `PATH`:

```bash
work_queue_worker --version
```

If `work_queue_worker` is missing, `run-series --forcebalance-workers N` exits with a
clear error telling you to install `ndcctools`. (You can omit `--forcebalance-workers`
to run ForceBalance serially without cctools.)

> **Note:** This setup runs the manager and all workers on **one machine** (localhost).
> Work Queue can also distribute workers across a cluster (via a shared port or a project
> name through a catalog server); that is out of scope here — ask if you need it.

---

## The command

A full joint fit of a congeneric series, with xTB drives, DFT single points, and a
parallel ForceBalance fit:

```bash
openff-bespoke executor run-series \
  --joint \
  --file ligands.sdf \
  --workflow default \
  --default-qc-spec xtb gfn2xtb none \
  --single-point-qc-spec psi4 b3lyp-d3bj dzvp \
  --base-tmd-ff project_base.py \
  --output bespoke_tmd_ff.py \
  --forcebalance-workers 16 \
  --n-fragmenter-workers 3 \
  --n-qc-compute-workers 3 \
  --qc-compute-n-cores 8 \
  --n-optimizer-workers 3
```

This will:

1. launch a temporary bespoke executor,
2. fragment and run an xTB torsion drive + DFT single point for every molecule,
3. pool every molecule's torsion data and run **one** ForceBalance fit (parallelized
   across 16 workers) of the base force field's shared torsion SMIRKS,
4. merge the fit torsions onto `project_base.py` and write `bespoke_tmd_ff.py`.

`bespoke_tmd_ff.py` is a tmd force-field dict you can load in Timemachine for the FEP map.

### Inputs

- `--file ligands.sdf` — an SDF (or other file) containing your series. May be given
  multiple times, or use `--smiles "..."` (repeatable) instead.
- `--workflow default` — required (or `--workflow-file your.json`).

### QC level of theory

- `--default-qc-spec PROGRAM METHOD BASIS` — the level the **torsion drive geometries**
  are optimized at, e.g. `xtb gfn2xtb none` (use `none` for the basis when not needed).
- `--single-point-qc-spec PROGRAM METHOD BASIS` — optional; the level the **grid
  energies** are recomputed at on the xTB geometries, e.g. `psi4 b3lyp-d3bj dzvp`. Omit
  to fit against the drive energies directly.

### Force field

- `--force-field` — the OpenFF force field the fits start from. **Defaults to
  `openff_unconstrained-2.0.0.offxml`** to match a 2.0.0-derived tmd base. If your base
  derives from a different OpenFF version, set this to match (see "Force-field lineage"
  below).
- `--base-tmd-ff project_base.py` — required; the tmd force field to merge onto.
- `--output bespoke_tmd_ff.py` — where to write the combined tmd force field.

### Fitting mode

|Flag|Effect|
|-|-|
|`--joint`|One shared fit across the whole series (recommended for an FEP map).|
|`--per-molecule`|(default) Fit each molecule independently and merge its bespoke torsions.|

---

## Parallelism & resource configuration

The run has **two phases that do not overlap**, so you can size each for the whole
machine:

**Phase A — QC generation (executor running).** Dominated by the torsion drives + DFT
single points.

|Flag|Meaning|
|-|-|
|`--n-qc-compute-workers`|Parallel QC workers (more = more molecules/torsions at once).|
|`--qc-compute-n-cores`|Cores per QC worker. Total cores ≈ workers × cores.|
|`--qc-compute-max-mem`|GB per core per QC worker.|
|`--n-fragmenter-workers`|Parallel fragmentation workers.|
|`--n-optimizer-workers`|Parallel per-molecule fits (see caveat below).|

**Phase B — the joint ForceBalance fit (executor shut down).** One ForceBalance process,
parallelized internally:

|Flag|Meaning|
|-|-|
|`--forcebalance-workers`|cctools workers (one core each) evaluating fitting targets concurrently. Size up to your core count.|
|`--forcebalance-wq-port`|Work Queue port (defaults to an automatically chosen free port).|

The joint fit runs **after** the executor (and its workers) shut down, so
`--forcebalance-workers` can use the full machine without contending with the QC workers.

> **Why ForceBalance is the slow step, and why this helps.** Each ForceBalance iteration
> computes a finite-difference gradient: for *P* fitted parameters and *T* targets it
> performs roughly `(2P+1)·T` target evaluations. In a joint fit *T* (every molecule's
> torsions) and *P* (the shared SMIRKS) are both large, so this is the bottleneck. Work
> Queue spreads those evaluations across workers.

---

## Force-field lineage (important)

A joint fit refits the base force field's **generic** torsion SMIRKS and then **overrides**
the matching torsions in your tmd base. For that override to land, the SMIRKS strings of
the fit force field must match those in the tmd base — i.e. they must come from the **same
OpenFF lineage**.

Before launching, `run-series` checks the overlap and **warns** if fewer than 90% of the
fit force field's torsion SMIRKS are present in the base tmd force field:

```
! only 71% of the initial force field's torsion SMIRKS (openff_unconstrained-2.2.0.offxml)
  are present in the base tmd force field. Fit torsions whose SMIRKS are absent will be
  appended rather than overriding the base — if the base derives from a different OpenFF
  version, set --force-field to match its lineage.
```

If you see this, set `--force-field` to the OpenFF version your tmd base was built from
(e.g. `openff_unconstrained-2.0.0.offxml`).

At the end, the command reports how the merge landed:

```
✓ appended 0 and overrode 58 torsion(s) in the base tmd force field
```

For a clean joint fit you want **mostly "overrode"** — lots of "appended" means a lineage
mismatch.

---

## How it works under the hood

1. Each molecule is submitted to the executor with **broad SMIRKS**
   (`generate_bespoke_terms=False`), so the molecules share the base force field's generic
   torsion patterns. The executor generates (and caches) the xTB drives + DFT single
   points and runs a per-molecule fit.
2. For the joint fit, each completed result carries its populated QC reference data; these
   are pooled into a single ForceBalance optimization whose **parameters** are the
   de-duplicated shared SMIRKS and whose **targets** are every molecule's torsion profile.
3. That one fit runs (optionally Work-Queue-parallelized) and produces a refit force field.
4. The torsions that changed are merged onto the tmd base — overriding matching SMIRKS,
   appending new ones — and written out.

---

## When a DFT single point fails to converge

Single-point energies are computed **per grid point and are fault tolerant**. If a grid
point's DFT single point fails to converge:

- that grid point is **dropped** from the torsion profile (both its energy and geometry),
  with a warning logged by the QC worker — the rest of the drive is kept;
- the molecule is only **failed entirely** if fewer than **50%** of its grid points
  converge (a signal that something is systematically wrong for that molecule);
- a failed molecule is skipped (and, in `--per-molecule` mode, simply omitted from the
  merge; in `--joint` mode, omitted from the pooled fit). If *every* molecule fails the
  run aborts.

At the end of the run you get a single summary totalling the dropped points across the
whole series, e.g.:

```
! single-point summary: 7/168 DFT single points failed to converge and were dropped, across 4 torsion drive(s)
```

or, when everything converged:

```
✓ single-point summary: all 168 DFT single points converged
```

A handful of dropped points is normal and harmless (ForceBalance fits whatever grid
points are present). A large number, or dropped molecules, is worth investigating — try a
more robust functional/basis or SCF settings.

---

## Caveats

- **Redundant per-molecule fits.** In `--joint` mode the executor still runs a (discarded)
  per-molecule ForceBalance fit for each molecule — that is how the generated QC reference
  data is produced and returned for pooling. `--n-optimizer-workers` parallelizes those;
  the *real* fit is the single joint one, parallelized by `--forcebalance-workers`. The QC
  drives dominate cost, so this overhead is usually minor.
- **Validate on a few ligands first.** Run on 2–3 ligands and check the
  "overrode N / appended M" line and that the output loads in Timemachine before launching
  the full map.
- **cctools is only required for `--forcebalance-workers`.** Without it, omit the flag and
  ForceBalance runs serially.
