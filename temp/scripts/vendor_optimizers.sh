#!/usr/bin/env bash
# Vendors single-file optimizer implementations into optimizers/vendored/.
# URLs verified 2026-06. If an upstream moves, see the repo links in README.
set -e
cd "$(dirname "$0")/../optimizers/vendored"

curl -fsSL -o muon.py \
  https://raw.githubusercontent.com/KellerJordan/Muon/master/muon.py
curl -fsSL -o soap.py \
  https://raw.githubusercontent.com/nikhilvyas/SOAP/main/soap.py
curl -fsSL -o sophia.py \
  https://raw.githubusercontent.com/Liuhong99/Sophia/main/sophia.py

echo "vendored: $(ls *.py)"
