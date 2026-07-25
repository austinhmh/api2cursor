#!/usr/bin/env bash

set -euo pipefail

IMAGE_REFERENCE="${1:-}"
COMPOSE_FILE="${A2C_COMPOSE_FILE:-docker-compose.yml}"
SERVICE_NAME="${A2C_SERVICE_NAME:-api2cursor}"
CONTAINER_NAME="${A2C_CONTAINER_NAME:-api2cursor}"
ROLLBACK_IMAGE="api2cursor:rollback-previous"
HEALTH_URL="${A2C_HEALTH_URL:-http://127.0.0.1:3029/health}"

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
  GHCR_TOKEN="${GHCR_TOKEN:-${GH_TOKEN:-${GITHUB_TOKEN:-}}}"
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

replace_running_container() {
  local target_image="$1"
  if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    docker stop "${CONTAINER_NAME}" >/dev/null || true
    docker rm "${CONTAINER_NAME}" >/dev/null || true
  fi
  export API2CURSOR_IMAGE="${target_image}"
  "${COMPOSE_COMMAND[@]}" -f "${COMPOSE_FILE}" up -d --no-build --force-recreate "${SERVICE_NAME}"
}

replace_running_container "${IMAGE_REFERENCE}"

echo "Waiting for health: ${HEALTH_URL}"
for i in $(seq 1 30); do
  if curl -fsS "${HEALTH_URL}" >/tmp/api2cursor-health.json; then
    cat /tmp/api2cursor-health.json
    echo
    echo "Deployed ${IMAGE_REFERENCE}"
    exit 0
  fi
  sleep 2
done

echo "Health check failed; rolling back if possible."
if docker image inspect "${ROLLBACK_IMAGE}" >/dev/null 2>&1; then
  replace_running_container "${ROLLBACK_IMAGE}"
fi
exit 1
