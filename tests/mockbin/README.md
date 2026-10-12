Mock Slurm / nvidia-smi `sh` scripts for shell-integration tests (DESIGN.md §10.4). Owner: WP2.

Each script prints the fixture of `tests/fixtures/samples/$RF_MOCK_HOST/` that matches the arguments it receives
(translating the fixtures' `|` into the separator the command asked for) and appends its argv to `$RF_MOCK_LOG`
(one line per call, arguments joined by TAB; Slurm mocks add an `ENV` line with the Slurm variables they saw).
`_lib.sh` documents the knobs (`RF_MOCK_FILE_*` overrides, `RF_MOCK_*_ERR` failures, `RF_MOCK_NVIDIA_HANG`).

Tests link the commands they need into a temporary directory placed first on `PATH` (`tests/integration/harness.py`)
and set `RF_MOCK_BIN` to this directory. `ps`/`kill` are the real ones: the §6.2 signal-script test runs against real
throwaway processes it spawns itself.
