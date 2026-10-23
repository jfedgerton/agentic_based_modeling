#!/usr/bin/env bash
# Regenerates every derived file, figure and table from the raw simulation runs.
# Usage, from the repository root:   bash run_all.sh
# Raw runs are read from ABM_ROOT (default: two folders up, i.e. ABM/ when this repo is ABM/paper/llm_abm_pipeline).
set -euo pipefail
cd "$(dirname "$0")"
export PIPELINE_ROOT="$PWD"
export ABM_ROOT="${ABM_ROOT:-$(cd ../.. && pwd)}"
RUNS="$ABM_ROOT/five_simulation/PD_CV_outputs"
OUT="derived/text"
test -d "$RUNS" || { echo "Raw runs not found at $RUNS -- set ABM_ROOT=/path/to/ABM"; exit 1; }
mkdir -p "$OUT/per_run" logs figures tables
python3 -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9+ needed"'
python3 --version | tee logs/text_analysis.log

echo "== 1/2 Text analysis of agent rationales (Python, standard library only)"
rm -f "$OUT"/per_run/*.json
for d in "$RUNS"/pd_calculator_* "$RUNS"/pd_reasoner_* "$RUNS"/pd_roleplayer_* \
         "$RUNS"/cv_calculator_* "$RUNS"/cv_reasoner_* "$RUNS"/cv_roleplayer_*; do
  python3 python/01_rationales_by_run.py "$d" "$OUT" >> logs/text_analysis.log
done
python3 python/03_label_deviations.py "$RUNS" "$OUT" python/pd_rationale_codes.csv | tee -a logs/text_analysis.log
python3 python/02_summarise.py "$OUT" python/pd_rationale_codes.csv | tee -a logs/text_analysis.log
# The validation sample (python/validation/) is drawn once and hand-coded, so it is NOT redrawn here; only its machine
# labels are refreshed. To redraw it: python3 python/04_validation_sample.py derived/text/deviation_labels.csv.gz python/validation 50
python3 python/04_validation_sample.py "$OUT/deviation_labels.csv.gz" python/validation --rekey | tee -a logs/text_analysis.log
# Crisis game: reasons for overstating strength (Fig. 5B), and its validation sample (drawn once; rekeyed after that)
python3 python/05_ic_signal_reasons.py "$ABM_ROOT" "$OUT" | tee -a logs/text_analysis.log
if [ -f python/validation/ic_validation_key.csv ]; then
  python3 python/06_ic_validation_sample.py "$OUT/ic_signal_labels.csv.gz" python/validation --rekey | tee -a logs/text_analysis.log
else
  python3 python/06_ic_validation_sample.py "$OUT/ic_signal_labels.csv.gz" python/validation | tee -a logs/text_analysis.log
fi

echo "== 2/2 Statistics, figures and tables (R)"
Rscript R/run_all.R 2>&1 | tee logs/run_all_R.log
