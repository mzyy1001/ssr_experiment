#!/usr/bin/env bash
# auto_finalize.sh — runs locally, after seeds 456 and 789 land on chen.
#
# 1. SCP all seed shards down
# 2. Aggregate via finalize_table1_multiseed.py
# 3. Recompile paper_position_v2.pdf
# 4. git add + commit + push
#
# Idempotent: safe to re-run.
set -u
REPO=/home/mzyy1001/business
SSH="ssh -i $HOME/.ssh/id_server_chen -p 2222 -o StrictHostKeyChecking=no chenhongrui@122.225.39.134"
SCP="scp -i $HOME/.ssh/id_server_chen -P 2222 -o StrictHostKeyChecking=no"

echo ">>> [auto_finalize] pulling shards from chen"
$SCP "chenhongrui@122.225.39.134:/data/chenhongrui/business/results/table1/seed*_s*_shard*.json" "$REPO/results/table1/" 2>&1 | tail -3 || true

echo ">>> [auto_finalize] running finalize_table1_multiseed.py"
cd "$REPO" && python3 experiments/finalize_table1_multiseed.py 2>&1 | tail -25

echo ">>> [auto_finalize] recompiling paper_position_v2.pdf"
cd "$REPO/nips_src"
/usr/bin/pdflatex -interaction=nonstopmode -halt-on-error paper_position_v2.tex > /tmp/pdflatex_auto.log 2>&1
if [ ! -f paper_position_v2.pdf ]; then
    echo "PDFLATEX_FAILED"
    tail -20 /tmp/pdflatex_auto.log
    exit 1
fi
rm -f paper_position_v2.aux paper_position_v2.log paper_position_v2.out paper_position_v2.synctex.gz
echo "PDF: $(stat -c %s paper_position_v2.pdf) bytes"

echo ">>> [auto_finalize] committing + pushing"
cd "$REPO"
git add -A
if git diff --cached --quiet; then
    echo "AUTO_FINALIZE_NOTHING_TO_COMMIT"
    exit 0
fi
git commit -m "$(cat <<'EOF'
Multi-seed finalization: 4-seed mean ± std on 13Q benchmark

Auto-aggregated seeds 42/123/456/789 for the LLM-row cells (m1, c2)
and pulled per-seed shards into results/table1/. Patched Table 2 of
paper_position_v2.tex with multi-seed mean ± across-question std,
and recompiled paper_position_v2.pdf.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "AUTO_FINALIZE_DONE"
