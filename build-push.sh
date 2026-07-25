#!/usr/bin/env bash

set -euo pipefail

REMOTE_NAME="${A2C_BUILD_REMOTE:-origin}"
WORKFLOW_FILE="${A2C_BUILD_WORKFLOW:-ci-ghcr.yml}"
REPOSITORY="${A2C_BUILD_REPOSITORY:-austinhmh/api2cursor}"
IMAGE_NAME="${A2C_BUILD_IMAGE:-ghcr.io/austinhmh/api2cursor}"
RUN_DISCOVERY_TIMEOUT_SECONDS="${A2C_BUILD_RUN_DISCOVERY_TIMEOUT_SECONDS:-300}"
RUN_DISCOVERY_POLL_SECONDS="${A2C_BUILD_RUN_DISCOVERY_POLL_SECONDS:-15}"
RUN_STATUS_TIMEOUT_SECONDS="${A2C_BUILD_RUN_STATUS_TIMEOUT_SECONDS:-3600}"
RUN_STATUS_POLL_SECONDS="${A2C_BUILD_RUN_STATUS_POLL_SECONDS:-20}"

fail() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

command -v git >/dev/null 2>&1 || fail "git is required."
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "not inside a Git repository."

BRANCH_NAME="$(git branch --show-current)"
[[ -n "${BRANCH_NAME}" ]] || fail "detached HEAD is not supported."
if [[ "${BRANCH_NAME}" != "main" && "${BRANCH_NAME}" != feat/* ]]; then
  fail "branch ${BRANCH_NAME} does not trigger ${WORKFLOW_FILE}; use main or feat/**."
fi

git remote get-url "${REMOTE_NAME}" >/dev/null 2>&1 || fail "Git remote ${REMOTE_NAME} does not exist."

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  fail "tracked files are not clean; commit the intended changes before building."
fi

if [[ "${A2C_BUILD_ALLOW_UNTRACKED_SOURCE:-0}" != "1" ]]; then
  UNTRACKED_SOURCE_FILES="$(
    git ls-files --others --exclude-standard -- \
      '*.py' '*.sh' '*.yml' '*.yaml' 'Dockerfile*' 'tests/**'
  )"
  if [[ -n "${UNTRACKED_SOURCE_FILES}" ]]; then
    printf 'Error: untracked source files could be missing from the CI commit:\n%s\n' "${UNTRACKED_SOURCE_FILES}" >&2
    exit 1
  fi
fi

COMMIT_SHA="$(git rev-parse HEAD)"
[[ "${COMMIT_SHA}" =~ ^[0-9a-f]{40}$ ]] || fail "HEAD is not a full 40-character commit SHA."

USE_AUTHENTICATED_GH=false
if command -v gh >/dev/null 2>&1 && gh auth token --hostname github.com >/dev/null 2>&1; then
  USE_AUTHENTICATED_GH=true
fi
if [[ "${USE_AUTHENTICATED_GH}" != "true" ]]; then
  command -v curl >/dev/null 2>&1 || fail "curl is required when GitHub CLI is not authenticated."
  command -v jq >/dev/null 2>&1 || fail "jq is required when GitHub CLI is not authenticated."
fi

printf 'Pushing %s to %s (%s)...\n' "${COMMIT_SHA}" "${REMOTE_NAME}" "${BRANCH_NAME}" >&2
git push "${REMOTE_NAME}" "HEAD:refs/heads/${BRANCH_NAME}"

printf 'Waiting for workflow %s on %s...\n' "${WORKFLOW_FILE}" "${COMMIT_SHA}" >&2
START_TS="$(date +%s)"
RUN_ID=""
while true; do
  NOW="$(date +%s)"
  if (( NOW - START_TS > RUN_DISCOVERY_TIMEOUT_SECONDS )); then
    fail "timed out waiting for workflow run for ${COMMIT_SHA}."
  fi
  if [[ "${USE_AUTHENTICATED_GH}" == "true" ]]; then
    RUN_ID="$(
      gh run list \
        --repo "${REPOSITORY}" \
        --workflow "${WORKFLOW_FILE}" \
        --commit "${COMMIT_SHA}" \
        --limit 5 \
        --json databaseId,status,conclusion,headSha \
        --jq ".[] | select(.headSha==\"${COMMIT_SHA}\") | .databaseId" \
        2>/dev/null | head -n1 || true
    )"
  fi
  if [[ -n "${RUN_ID}" ]]; then
    break
  fi
  sleep "${RUN_DISCOVERY_POLL_SECONDS}"
done

printf 'Found run %s; waiting for completion...\n' "${RUN_ID}" >&2
if [[ "${USE_AUTHENTICATED_GH}" == "true" ]]; then
  gh run watch "${RUN_ID}" --repo "${REPOSITORY}" --exit-status
  CONCLUSION="$(gh run view "${RUN_ID}" --repo "${REPOSITORY}" --json conclusion --jq .conclusion)"
else
  fail "authenticated gh is required to watch the run."
fi

if [[ "${CONCLUSION}" != "success" ]]; then
  fail "workflow ${WORKFLOW_FILE} concluded with ${CONCLUSION}."
fi

printf '%s:sha-%s\n' "${IMAGE_NAME}" "${COMMIT_SHA}"
