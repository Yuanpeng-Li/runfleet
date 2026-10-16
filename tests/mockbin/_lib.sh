# Shared helpers of the mock Slurm / nvidia-smi commands (sourced, POSIX sh). DESIGN.md §10.4.
#
# Environment (RF_MOCK_* survives the probe prelude, which unsets SQUEUE_*/SACCT_*/...):
#   RF_MOCK_SAMPLES   tests/fixtures/samples       RF_MOCK_HOST  fixture host (k2, vega, lyra, lynx, ...)
#   RF_MOCK_LOG       file receiving one line per call: NAME<TAB>arg<TAB>arg... (+ an ENV line for Slurm mocks)
#   RF_MOCK_FILE_<X>  absolute path overriding fixture X (SQUEUE, SACCT, SINFO, SSTAT, GPU, APPS, SPRIO, SSHARE)
#   RF_MOCK_<CMD>_ERR make <cmd> print this to stderr and exit 1 (SQUEUE, SACCT, SCONTROL, SINFO, SSTAT, SCANCEL,
#                     SPRIO, SSHARE)
: "${RF_MOCK_SAMPLES:?RF_MOCK_SAMPLES not set}" "${RF_MOCK_HOST:?RF_MOCK_HOST not set}"
S="$RF_MOCK_SAMPLES/$RF_MOCK_HOST"

mock_log() {
  [ -n "${RF_MOCK_LOG-}" ] || return 0
  { printf '%s' "$1"; shift; for a in "$@"; do printf '\t%s' "$a"; done; printf '\n'; } >>"$RF_MOCK_LOG"
}

mock_env() {
  [ -n "${RF_MOCK_LOG-}" ] || return 0
  printf 'ENV\t%s\tSLURM_TIME_FORMAT=%s\tSQUEUE_FORMAT=%s\tSQUEUE_STATES=%s\tSACCT_FORMAT=%s\tSINFO_FORMAT=%s\tSLURM_CLUSTERS=%s\n' \
    "$1" "${SLURM_TIME_FORMAT-}" "${SQUEUE_FORMAT-}" "${SQUEUE_STATES-}" "${SACCT_FORMAT-}" "${SINFO_FORMAT-}" \
    "${SLURM_CLUSTERS-}" >>"$RF_MOCK_LOG"
}

mock_fail() {   # mock_fail NAME: honour RF_MOCK_<NAME>_ERR
  eval "rf_err=\${RF_MOCK_${1}_ERR-}"
  if [ -n "$rf_err" ]; then printf '%s\n' "$rf_err" >&2; exit 1; fi
}

fixture() {     # fixture NAME DEFAULT_SAMPLE: path of the fixture to print
  eval "rf_f=\${RF_MOCK_FILE_${1}-}"
  if [ -n "$rf_f" ]; then printf '%s' "$rf_f"; else printf '%s' "$S/$2.txt"; fi
}

emit() {        # emit FILE SEP: print a fixture, translating its "|" separators into SEP
  if [ -z "$2" ] || [ "$2" = "|" ]; then cat "$1"; else tr '|' "$2" <"$1"; fi
}

first_char() { rf_rest=$1; printf '%s' "${rf_rest%"${rf_rest#?}"}"; }
