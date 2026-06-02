#!/usr/bin/env bash
# Generate a custom bespokefit workflow file from the built-in `default` workflow with a
# tweaked WBO fragmentation threshold. A larger threshold -> the fragmenter stops growing
# sooner -> smaller fragments (more torsions shared across a congeneric series), at the
# cost of fragment electronics that less faithfully reproduce the parent.
#
# Pass the result to run-series with --workflow-file (NOT --workflow; they are mutually
# exclusive). NOTE: changing the threshold changes the QM torsion drives, so it
# invalidates the QC cache for affected torsions -- they will be recomputed.
#
# Usage:
#   ./make_workflow.sh 0.1                       # -> wbo_thresh_0.1.json
#   ./make_workflow.sh 0.05 my_workflow.json     # custom output name
#   BASE_WORKFLOW=debug ./make_workflow.sh 0.1   # base off a different built-in workflow
set -euo pipefail

THRESHOLD="${1:-}"
BASE_WORKFLOW="${BASE_WORKFLOW:-default}"

if [[ -z "$THRESHOLD" ]]; then
    echo "usage: $0 <threshold> [output.json]   (e.g. $0 0.1)" >&2
    echo "  default fragmenter threshold is 0.03; larger -> smaller fragments" >&2
    exit 2
fi
OUTPUT="${2:-wbo_thresh_${THRESHOLD}.json}"

THRESHOLD="$THRESHOLD" BASE_WORKFLOW="$BASE_WORKFLOW" OUTPUT="$OUTPUT" python - <<'EOF'
import json
import os

from openff.utilities import get_data_file_path

threshold = float(os.environ["THRESHOLD"])
base = os.environ["BASE_WORKFLOW"]
output = os.environ["OUTPUT"]

src = get_data_file_path(os.path.join("schemas", f"{base.lower()}.json"),
                         "openff.bespokefit")
workflow = json.load(open(src))

engine = workflow.get("fragmentation_engine")
if engine is None:
    raise SystemExit(f"workflow '{base}' has no fragmentation_engine to tweak")
if engine.get("scheme") != "WBO":
    raise SystemExit(f"workflow '{base}' fragmenter scheme is {engine.get('scheme')!r}, "
                     "not WBO; threshold only applies to the WBO fragmenter")

old = engine.get("threshold")
engine["threshold"] = threshold
json.dump(workflow, open(output, "w"), indent=2)
print(f"wrote {output}: WBO threshold {old} -> {threshold} (base workflow: {base})")
EOF

echo
echo "use it with:"
echo "  openff-bespoke executor run-series --workflow-file $OUTPUT --hybrid ...your other flags..."
echo "(preview the effect on sharing first:  WORKFLOW_FILE=$OUTPUT ./run_torsion_sharing.sh series.sdf)"
