#!/bin/bash
# Wake up autofs/CIFS mounts before starting Docker
cd "$(dirname "$0")"

echo "Activating network mounts..."
ls /home/nepomuk/sflnas/DataReadWrite334/0_shared/Feldschwarm/bagseek/src > /dev/null 2>&1
ls /home/nepomuk/sflnas/DataReadOnly334/tractor_data/autorecord > /dev/null 2>&1

echo "Starting containers..."
docker compose down 2>/dev/null

# Check if --build flag is passed (in any position)
BUILD_FLAG=false
SHOW_LOGS=false

for arg in "$@"; do
    if [[ "$arg" == "--build" ]]; then
        BUILD_FLAG=true
    elif [[ "$arg" == "--logs" ]]; then
        SHOW_LOGS=true
    fi
done

if [[ "$BUILD_FLAG" == "true" ]]; then
    docker compose up -d --build
else
    docker compose up -d
fi

echo "Waiting for healthy status..."
docker compose ps

echo "Showing logs..."
if [[ "$SHOW_LOGS" == "true" ]]; then
    docker compose logs -f
fi