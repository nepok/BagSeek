#!/bin/bash

SESSION_NAME="bagseek-frontend"

# Starte neue tmux-Session oder überschreibe alte
tmux kill-session -t $SESSION_NAME 2>/dev/null
tmux new-session -d -s $SESSION_NAME -n Frontend

# Frontend starten
tmux send-keys -t $SESSION_NAME:Frontend 'cd /mnt/data/bagseek/react-frontend' C-m
tmux send-keys -t $SESSION_NAME:Frontend 'npm start' C-m

# Öffne tmux-Session
tmux attach-session -t $SESSION_NAME