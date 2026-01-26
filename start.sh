#!/bin/bash
# Wake up autofs/CIFS mounts before starting Docker
cd "$(dirname "$0")"

echo "Activating network mounts..."
ls /home/nepomuk/sflnas/DataReadWrite334/0_shared/Feldschwarm/bagseek/src > /dev/null 2>&1
ls /home/nepomuk/sflnas/DataReadOnly334/tractor_data/autorecord > /dev/null 2>&1

echo "Starting containers..."
docker compose down 2>/dev/null
docker compose up -d

echo "Waiting for healthy status..."
docker compose ps
