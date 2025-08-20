#!/bin/bash

SESSION_NAME="bagseek-dev"

# Start tmux session
tmux new-session -d -s $SESSION_NAME

# Fenster 1: Backend
tmux rename-window -t $SESSION_NAME:0 'Backend'
tmux send-keys -t $SESSION_NAME:Backend 'cd /mnt/data/bagseek/flask-backend/api' C-m
tmux send-keys -t $SESSION_NAME:Backend 'conda activate bagseek-gpu' C-m
tmux send-keys -t $SESSION_NAME:Backend 'flask run --debug' C-m

# Fenster 2: Frontend
tmux new-window -t $SESSION_NAME:1 -n 'Frontend'
tmux send-keys -t $SESSION_NAME:Frontend 'cd /mnt/data/bagseek/react-frontend' C-m
tmux send-keys -t $SESSION_NAME:Frontend 'npm start' C-m

# Öffne tmux-Session
tmux attach -t $SESSION_NAME