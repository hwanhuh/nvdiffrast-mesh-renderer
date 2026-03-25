#!/usr/bin/env bash
set -euo pipefail

NVDIFFRAST_REF="${NVDIFFRAST_REF:-v0.4.0}"
NVDIFFRAST_REPO="${NVDIFFRAST_REPO:-https://github.com/NVlabs/nvdiffrast.git}"
NVDIFFRAST_CLONE_DIR="${NVDIFFRAST_CLONE_DIR:-/tmp/extensions/nvdiffrast}"

if [[ -d "${NVDIFFRAST_CLONE_DIR}/.git" ]]; then
    git -C "${NVDIFFRAST_CLONE_DIR}" fetch --tags origin
    git -C "${NVDIFFRAST_CLONE_DIR}" checkout "${NVDIFFRAST_REF}"
else
    mkdir -p "$(dirname "${NVDIFFRAST_CLONE_DIR}")"
    git clone -b "${NVDIFFRAST_REF}" "${NVDIFFRAST_REPO}" "${NVDIFFRAST_CLONE_DIR}"
fi

python -m pip install "${NVDIFFRAST_CLONE_DIR}" --no-build-isolation
