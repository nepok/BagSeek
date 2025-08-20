#!/bin/bash

SESSION_NAME="bagseek-backend"

# Starte neue tmux-Session oder überschreibe alte
tmux kill-session -t $SESSION_NAME 2>/dev/null
tmux new-session -d -s $SESSION_NAME -n Backend

# Backend starten
tmux send-keys -t $SESSION_NAME:Backend 'cd /mnt/data/bagseek/flask-backend/api' C-m
tmux send-keys -t $SESSION_NAME:Backend 'conda activate bagseek-gpu' C-m
tmux send-keys -t $SESSION_NAME:Backend 'flask run --debug' C-m

# Öffne tmux-Session
tmux attach-session -t $SESSION_NAME