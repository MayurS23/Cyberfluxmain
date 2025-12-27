#!/bin/bash

# Cyberflux Quick Start Script

echo "====================================="
echo "  CYBERFLUX QUICK START"
echo "====================================="
echo ""

echo "[1/3] Checking services..."
sudo supervisorctl status

echo ""
echo "[2/3] Testing backend API..."
curl -s ${REACT_APP_BACKEND_URL}/api/ | python3 -c "import sys, json; print(json.load(sys.stdin)['message'])"

echo ""
echo "[3/3] System ready!"
echo ""
echo "Dashboard: Check your browser"
echo "Backend API: ${REACT_APP_BACKEND_URL}/api"
echo ""
echo "To train ML models: bash /app/scripts/train_models.sh"
echo ""
