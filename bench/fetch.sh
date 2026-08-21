#!/usr/bin/env bash
# Fetch the benchmark suites used in Table I.
#   ISCAS'85 / ISCAS'89 / ITC'99 / MCNC
# Circuits land in bench/circuits/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p circuits && cd circuits

echo "== EPFL benchmark mirror (ISCAS + MCNC in BLIF) =="
if [ ! -d benchmarks ]; then
  git clone --depth 1 https://github.com/lsils/benchmarks.git
fi

echo "== ITC'99 (b15, b17) =="
if [ ! -d itc99 ]; then
  mkdir -p itc99
  echo "  ITC'99 sources vary by mirror; place b15.bench and b17.bench here."
fi

cat <<'NOTE'

Table I circuits and where they come from:
  ISCAS'85  c17 c1355 c1908 c2670 c7552
  ISCAS'89  s386 s526 s713 s1494 s5378
  MCNC      i8 i10 des
  ITC'99    b15 b17

The flow expects .bench or .blif input; conversion to .th needs the ABC
driver (src/tlglock/abc.py), which is not written yet.
NOTE
