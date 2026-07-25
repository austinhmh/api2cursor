#!/usr/bin/env bash
set -euo pipefail

REMOTE_NAME="${A2C_BUILD_REMOTE:-origin}"
WORKFLOW_FILE="${A2C_BUILD_WORKFLOW:-ci-ghcr.yml}"
REPOSITORY="${A2C_BUILD_REPOSITORY:-austinhmh/api2cursor}"
IMAGE_NAME="${A2C_BUILD_IMAGE:-ghcr.io/austinhmh/api2cursor}"
RUN_DISCOVERY_TIMEOUT_SECONDS="${A2C_BUILD_RUN_DISCOVERY_TIMEOUT_SECONDS:-300}"
RUN_DISCOVERY_POLL_SECONDS="${A2C_BUILD_RUN_DISCOVERY_POLL_SECONDS:-15}"

fail() { printf 'Error: %s\n' "$1" >&2; exit 1; }

command -v git >/dev/null 2>&1 || fail "git is required."
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "not inside a Git repository."

BRANCH_NAME="$(git branch --show-current)"
[[ -n "${BRANCH_NAME}" ]] || fail "detached HEAD is not supported."
if [[ "${BRANCH_NAME}" != "main" && "${BRANCH_NAME}" != feat/* ]]; then
  fail "branch ${BRANCH_NAME} does not trigger ${WORKFLOW_FILE}; use main or feat/**."
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  fail "tracked files are not clean; commit the intended changes before building."
fi

COMMIT_SHA="$(git rev-parse HEAD)"
[[ "${COMMIT_SHA}" =~ ^[0-9a-f]{40}$ ]] || fail "HEAD is not a full 40-character commit SHA."

command -v gh >/dev/null 2>&1 || fail "GitHub CLI (gh) is required."
gh auth token --hostname github.com >/dev/null 2>&1 || fail "gh is not authenticated."

git push "${REMOTE_NAME}" "HEAD:refs/heads/${BRANCH_NAME}" >&2

RUN_ID=""
DISCOVERY_DEADLINE=$((SECONDS + RUN_DISCOVERY_TIMEOUT_SECONDS))
while (( SECONDS < DISCOVERY_DEADLINE )); do
  RUN_ID="$(
    gh run list \
      --repo "${REPOSITORY}" \
      --workflow "${WORKFLOW_FILE}" \
      --branch "${BRANCH_NAME}" \
      --event push \
      --limit 30 \
      --json databaseId,headSha \
      --jq "map(select(.headSha == \"${COMMIT_SHA}\"))[0].databaseId // empty"
  )"
  if [[ -n "${RUN_ID}" ]]; then
    break
  fi
  sleep "${RUN_DISCOVERY_POLL_SECONDS}"
done
[[ -n "${RUN_ID}" ]] || fail "timed out waiting for ${WORKFLOW_FILE} at ${COMMIT_SHA}."

gh run watch "${RUN_ID}" --repo "${REPOSITORY}" --exit-status >&2

RUN_CONCLUSION="$(
  gh run view "${RUN_ID}" --repo "${REPOSITORY}" --json conclusion --jq .conclusion
)"
[[ "${RUN_CONCLUSION}" == "success" ]] || fail "workflow conclusion ${RUN_CONCLUSION} is not success."

printf '%s:sha-%s\n' "${IMAGE_NAME}" "${COMMIT_SHA}"
