#!/usr/bin/env bash
# tmux helper to run training in background + monitor.
set -euo pipefail

if ! command -v tmux >/dev/null; then
    echo "Installing tmux ..."
    apt-get install -y tmux 2>/dev/null || sudo apt-get install -y tmux
fi

SESSION="fastpretrain"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n "train"
tmux send-keys -t "$SESSION:train" \
    "cd $SCRIPT_DIR && bash 02_train.sh 2>&1 | tee /workspace/fast_pretrain_output/train.log" C-m

tmux new-window -t "$SESSION" -n "gpu"
tmux send-keys -t "$SESSION:gpu" \
    "watch -n 5 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'" C-m

tmux new-window -t "$SESSION" -n "log"
tmux send-keys -t "$SESSION:log" "sleep 30 && tail -F /workspace/fast_pretrain_output/train.log" C-m

tmux new-window -t "$SESSION" -n "shell"

echo "tmux session 'fastpretrain' started."
echo "Attach: tmux attach -t fastpretrain"
echo "Detach: Ctrl-B then d"
echo "Switch windows: Ctrl-B then 0/1/2/3"
