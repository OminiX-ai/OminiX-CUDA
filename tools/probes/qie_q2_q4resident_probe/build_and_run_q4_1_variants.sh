#!/usr/bin/env bash
# QIE-Q2.2 Q4_1 variant probe — build + run on ac03.
# Tests 3 sign/layout hypotheses in parallel against the shared CPU
# reference. See test_qie_q4_1_variants.cpp header for the rationale.
#
# Reproduction:  bash build_and_run_q4_1_variants.sh
# Exit codes:
#   0 = GREEN  (at least one variant cleared cos_sim > 0.99)
#   1 = YELLOW (best variant 0.90 < cos_sim <= 0.99)
#   2 = RED    (all variants failed — escalate per smoke §7)
set -euo pipefail

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}
source $ASCEND_TOOLKIT_HOME/../set_env.sh 2>/dev/null || true

DIR=$(cd "$(dirname "$0")" && pwd)
cd "$DIR"

# Cohabit with Agent A4c — honour the ac03 HBM lock.
LOCK=/tmp/ac03_hbm_lock
if [ -e "$LOCK" ]; then
    echo "[probe] HBM lock present at $LOCK — waiting..."
    while [ -e "$LOCK" ]; do sleep 5; done
fi
echo "qie_q2_q4_1_variants $$" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

g++ -std=c++17 -O2 -o test_qie_q4_1_variants test_qie_q4_1_variants.cpp \
    -I$ASCEND_TOOLKIT_HOME/aarch64-linux/include \
    -L$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64 \
    -lascendcl -lopapi -lnnopbase -ldl

echo "--- build OK ---"
./test_qie_q4_1_variants
rc=$?
echo "--- probe exit rc=$rc ---"
exit $rc
