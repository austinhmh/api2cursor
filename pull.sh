#!/usr/bin/env bash
set -euo pipefail

IMAGE_REFERENCE="${1:-}"
COMPOSE_FILE="${A2C_COMPOSE_FILE:-docker-compose.yml}"
SERVICE_NAME="api2cursor"
CONTAINER_NAME="api2cursor"
ROLLBACK_IMAGE="api2cursor:rollback-previous"

if docker compose version >/dev/null 2>&1; then
  COMPOSE_COMMAND=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_COMMAND=(docker-compose)
else
  echo "Error: neither docker compose nor docker-compose is available."
  exit 1
fi

if [[ -z "${IMAGE_REFERENCE}" ]]; then
  echo "Usage: ./pull.sh ghcr.io/austinhmh/api2cursor:sha-<40-hex-commit>"
  exit 1
fi

if [[ ! "${IMAGE_REFERENCE}" =~ :sha-[0-9a-f]{40}$ ]] && [[ ! "${IMAGE_REFERENCE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "Error: only immutable full-commit tags or image digests are accepted."
  exit 1
fi

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "Error: compose file not found: ${COMPOSE_FILE}"
  exit 1
fi

if [[ "${IMAGE_REFERENCE}" == ghcr.io/* ]]; then
  GHCR_TOKEN="${GHCR_TOKEN:-${GH_TOKEN:-}}"
  GHCR_USERNAME="${GHCR_USERNAME:-austinhmh}"
  if [[ -n "${GHCR_TOKEN}" ]]; then
    printf '%s' "${GHCR_TOKEN}" | docker login ghcr.io --username "${GHCR_USERNAME}" --password-stdin >/dev/null
  fi
fi

CURRENT_IMAGE_ID="$(docker inspect --format '{{.Image}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
if [[ -n "${CURRENT_IMAGE_ID}" ]]; then
  docker image tag "${CURRENT_IMAGE_ID}" "${ROLLBACK_IMAGE}" || true
fi

echo "Pulling CI-built image: ${IMAGE_REFERENCE}"
docker pull "${IMAGE_REFERENCE}"
CANDIDATE_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE_REFERENCE}")"

TMP_COMPOSE="$(mktemp)"
trap 'rm -f "${TMP_COMPOSE}"' EXIT
python3 - <<PY
from pathlib import Path
import re
text = Path("${COMPOSE_FILE}").read_text()
image = """${IMAGE_REFERENCE}"""
text2, n = re.subn(r'(^\s*image:\s*).*$', r'\g<1>' + image, text, count=1, flags=re.M)
if n == 0:
    raise SystemExit('failed to rewrite image in compose file')
Path("${TMP_COMPOSE}").write_text(text2)
PY

replace_running_container() {
  local target_compose="$1"
  if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    docker stop "${CONTAINER_NAME}" >/dev/null || true
    docker rm "${CONTAINER_NAME}" >/dev/null || true
  fi
  "${COMPOSE_COMMAND[@]}" -f "${target_compose}" up -d --no-build --force-recreate "${SERVICE_NAME}"
}

if ! replace_running_container "${TMP_COMPOSE}"; then
  echo "Candidate failed to start."
  exit 1
fi

healthy=false
for attempt in $(seq 1 30); do
  if curl -fsS --max-time 2 http://127.0.0.1:3029/health >/dev/null; then
    healthy=true
    break
  fi
  sleep 2
done

if [[ "${healthy}" != "true" ]]; then
  echo "Health check failed."
  docker logs "${CONTAINER_NAME}" 2>&1 | tail -50 || true
  exit 1
fi

RUNNING_IMAGE_ID="$(docker inspect --format '{{.Image}}' "${CONTAINER_NAME}")"
if [[ "${RUNNING_IMAGE_ID}" != "${CANDIDATE_IMAGE_ID}" ]]; then
  echo "Running image mismatch: ${RUNNING_IMAGE_ID} != ${CANDIDATE_IMAGE_ID}"
  exit 1
fi

# Persist the image reference into the real compose file for next restarts.
python3 - <<PY
from pathlib import Path
import re
path = Path("${COMPOSE_FILE}")
text = path.read_text()
image = """${IMAGE_REFERENCE}"""
text2, n = re.subn(r'(^\s*image:\s*).*$', r'\g<1>' + image, text, count=1, flags=re.M)
if n:
    path.write_text(text2)
PY

echo "Deployment verified."
echo "Image: ${IMAGE_REFERENCE}"
echo "Image ID: ${CANDIDATE_IMAGE_ID}"
