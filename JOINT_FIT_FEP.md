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
  --default-qc-spec xtb gfn2xtb none or --default-qc-spec torchani ani2x none \
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
|`--joint`|One shared fit across the whole series using the base force field's broad SMIRKS (recommended starting point for an FEP map).|
|`--hybrid`|Pooled fit with **bespoke** SMIRKS: each unique fragment torsion fit once — shared scaffold torsions consistently, R-group torsions with their own bespoke parameters. Use when the diagnostic shows R-group torsions fitting worse than the scaffold. Takes precedence over `--joint`.|
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
|`--n-optimizer-workers`|Parallel per-molecule fits (`--per-molecule` only; skipped in `--joint`).|

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
   points; the per-molecule fit is **skipped** (mocked) so only the QC is produced.
2. For the joint fit, each completed result carries its populated QC reference data; these
   are pooled into a single ForceBalance optimization whose **parameters** are the
   de-duplicated shared SMIRKS and whose **targets** are every molecule's torsion profile.
3. That one fit runs (optionally Work-Queue-parallelized) and produces a refit force field.
4. The torsions that changed are merged onto the tmd base — overriding matching SMIRKS,
   appending new ones — and written out.

---

## Reusing shared torsions across the series

A congeneric series shares a scaffold, so most torsions recur across ligands. bespokefit
caches QC by a canonicalized task hash, so each **unique** torsion drive is computed once
and reused everywhere it appears — this is what keeps a whole series tractable in an FEP
timeframe. After the QC stage the run reports how much was shared:

```
✓ computed 23 unique torsion drives, reused 71 from cache (across 12 molecule(s))
```

For a tight congeneric set you want **unique ≪ total** — lots of "reused" means the shared
scaffold is being computed once, as intended. If "unique" is close to the total, the
fragments aren't matching across ligands (e.g. R-groups sit too close to the scaffold
torsions), and the series isn't sharing as much as it could.

In a `--joint` fit the same shared drive is referenced by every molecule that contains it;
those duplicate references are **de-duplicated by identity** (fragment CMILES + scanned
dihedral) before fitting, so a shared torsion is fit once against its data rather than
over-weighted — and ForceBalance doesn't waste target evaluations on the copies.

---

## Is the joint fit good enough? (torsion-fit diagnostic)

After a `--joint` fit (unless you pass `--no-diagnose-fit`), the fitted force field is
scored against the QM torsion drives — the RMSE between the MM and QM *relative* energy
profiles (single-point MM energies at the QM grid geometries) — and reported split by
**scaffold (recurring)** vs **R-group (unique)** torsions:

```
torsion-fit diagnostic (MM vs QM relative energies, kcal/mol):
    scaffold (recurring): 18 drives — mean 0.31, median 0.28, max 0.79
    R-group (unique)   :  7 drives — mean 1.12, median 0.95, max 2.40
    → R-group torsions fit ~3.6x worse than the scaffold — a bespoke R-group (hybrid)
      treatment would likely help.
```

This tells you whether the shared-SMIRKS joint fit reproduces the unique R-group torsions
as well as the shared scaffold ones. If the R-group RMSEs are comparable to the scaffold's,
the joint fit is adequate and you're done. If they're much larger, the generic SMIRKS
aren't capturing those R-group torsions and the `--hybrid` fit below is worth trying. The
diagnostic adds a short post-fit step and needs OpenMM; the recurring/unique split uses the
same shared-fragment identity as the de-duplication.

---

## The hybrid fit (`--hybrid`)

`--hybrid` keeps the consistency of a joint fit for the scaffold while giving each R-group
torsion its own bespoke parameter. It runs the series with **bespoke** SMIRKS, skips the
per-molecule fit, and then runs **one** pooled ForceBalance optimization in which:

- each **unique fragment torsion** (by fragment CMILES + scanned dihedral) is fit **once**;
- a torsion **shared** across the series (same fragment) therefore appears once and is fit
  consistently — the *scaffold-joint* half;
- a torsion **unique** to one molecule keeps its **own** bespoke parameter — the
  *R-group-bespoke* half.

Each scanned torsion is linked to its bespoke parameter by matching the parameter's SMIRKS
to the scanned central dihedral (ChemPer SMIRKS aren't guaranteed identical across
molecules, so they're pooled by fragment identity, not by SMIRKS string). The fit is run
against a merged force field — the base plus every molecule's bespoke parameters — and the
fitted bespoke torsions are then merged onto the tmd base. Because bespoke SMIRKS are more
specific than the base's generic patterns, they are **appended** (not string-overridden)
and win per-atom via SMIRNOFF's most-specific-match rule (so expect "appended N" rather
than "overrode N").

### How a torsion is classified as shared vs R-group

Classification is by **fragment identity** (fragment CMILES + scanned dihedral), and a
torsion counts as **shared the moment its fragment appears in ≥ 2 molecules** — there is
**no majority cutoff**. A torsion in 10/12, 3/12, or even 2/12 molecules is all treated the
same: shared. Only a torsion that appears in **exactly one** molecule is "unique" (R-group).

Crucially, the hybrid *fit itself applies no threshold* — it simply de-duplicates by
fragment identity, so a shared torsion's identical (cached) drives collapse to one and are
fit once (one consistent value applied to every molecule that has it), while a unique
torsion stays on its own. "Shared vs R-group" is therefore an emergent consequence of
de-duplication, not a branch in the code. The `≥ 2` threshold only labels the buckets in
the **torsion-fit diagnostic** (`recurring_threshold`, default 2); changing it re-labels
the report but does not change how anything is fit.

Two consequences worth knowing:

- It keys on the **fragment**, not the abstract chemical torsion. If 8 of those 12
  molecules produce an identical fragment but 2 have a slightly different local environment
  (an R-group close enough to bleed into the fragment), you get *two* identities — one
  shared by 8, one shared by 2 — and **both** are still "shared" (each ≥ 2). That's
  intentional: fragment recurrence captures whether the local chemistry is genuinely the
  same.
- A shared torsion need not appear in *all* molecules; the 2 molecules without it just
  don't carry that torsion, and the one fitted value applies to the 10 that do.

```
openff-bespoke executor run-series --hybrid \
  --file ligands.sdf --workflow default \
  --default-qc-spec xtb gfn2xtb none --single-point-qc-spec psi4 b3lyp-d3bj dzvp \
  --base-tmd-ff project_base.py --output bespoke_tmd_ff.py \
  --forcebalance-workers 48 --n-qc-compute-workers 1 --qc-compute-n-cores 48
```

**When to use it:** only when the torsion-fit diagnostic shows R-group torsions fitting
meaningfully worse than the scaffold under `--joint`. If they fit comparably, `--joint` is
simpler and sufficient. (`--hybrid` produces more, more-specific parameters and is more
sensitive to QC data quality for the unique torsions.)

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

- **No per-molecule fit in joint mode.** In `--joint` mode the per-molecule ForceBalance
  fit is **skipped** (mocked) — the executor only generates the QC, and a single joint fit
  over the whole series (parallelized by `--forcebalance-workers`) does the actual
  optimization. `--n-optimizer-workers` therefore has little effect with `--joint`; it
  matters for `--per-molecule` runs.
- **Validate on a few ligands first.** Run on 2–3 ligands and check the
  "overrode N / appended M" line and that the output loads in Timemachine before launching
  the full map.
- **cctools is only required for `--forcebalance-workers`.** Without it, omit the flag and
  ForceBalance runs serially.
