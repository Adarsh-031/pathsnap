#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

printf 'Which test partition should the server expose? [student_test/evaluator_test] ' >&2
if ! read -r partition; then
    printf '\nEnter student_test or evaluator_test to start the server.\n' >&2
    exit 1
fi

case "$partition" in
    student_test) data_root="$PWD/../Student folder" ;;
    evaluator_test) data_root="$PWD/.." ;;
    *)
        printf 'Invalid partition. Choose student_test or evaluator_test.\n' >&2
        exit 1
        ;;
esac

model="${PATHSNAP_MODEL:-$PWD/reports/model-tuning/logistic_regression_tuned.joblib}"
if [[ ! -d "$data_root/$partition" ]]; then
    printf 'Test partition directory not found: %s\n' "$data_root/$partition" >&2
    exit 1
fi
if [[ ! -f "$model" || ! -r "$model" ]]; then
    printf 'Model artifact is missing or unreadable: %s\n' "$model" >&2
    exit 1
fi

export PATHSNAP_TRIP_PARTITION="$partition"
export PATHSNAP_DATA_ROOT="$data_root"
export PATHSNAP_MODEL="$model"

printf 'Serving %s on port 8002.\n' "$partition"
exec uv run --locked --extra api uvicorn pathsnap.api:app --host 0.0.0.0 --port 8002
