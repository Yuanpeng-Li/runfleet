"""[WP4] `python -m runfleet` entry: delegates to cli.main()."""

from runfleet.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
