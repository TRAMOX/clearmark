# Contributing to ClearMark

Thanks for helping improve ClearMark. This guide covers how to set up a development environment, the standards we follow, and how changes get merged.

## Ways to contribute

- **Report a bug.** Open an issue using the *Bug report* template. Include your OS, Python version, the removal method used, and if possible a short clip or frame that reproduces the problem.
- **Suggest a feature.** Open an issue using the *Feature request* template and describe the problem it solves.
- **Submit code.** Fixes, performance work and new features are welcome. For larger changes, open an issue first so we can agree on the approach.

## Development setup

```bash
git clone https://github.com/TRAMOX/clearmark.git
cd clearmark
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest
```

Run the app with `python app.py` and the CLI with `python cli.py --help`.

## Coding standards

- Python 3.10+, standard library first. Add a dependency only when it pulls its weight.
- Match the surrounding style: type hints on public functions, short docstrings that explain *why*, and descriptive names.
- Keep the UI responsive. Anything slower than ~100 ms runs in a worker thread and reports back through the app's job queue.
- Processing code in `vwstudio/` must not import Tkinter, so it stays usable from the CLI and tests.
- Never modify a user's input file. Exports write to a temporary file and are moved into place only on success.

## Tests

- `tests/test_pipeline.py` builds a synthetic clip with a known logo and checks detection, removal quality and export. Add a test with every behaviour change.
- Removal-quality tests compare against the known clean frames. If you improve quality, tighten the thresholds.
- CI runs the tests on Windows and Ubuntu for every push and pull request.

## Pull requests

1. Fork the repository and create a branch from `main` (`feature/<short-name>` or `fix/<short-name>`).
2. Keep each pull request focused on one change, with a clear description of what and why.
3. Make sure `python -m pytest` passes locally.
4. Update `README.md` and `CHANGELOG.md` when behaviour or options change.
5. A maintainer will review it. Please respond to feedback within a reasonable time so the PR doesn't go stale.

## Commit messages

Use the imperative mood with a short summary line, for example `Add DirectML provider for AI fill`. Explain the reasoning in the body when it isn't obvious.

## Responsible use

Contributions that exist mainly to strip attribution from other people's content, or to defeat provenance and disclosure systems, won't be accepted. See the *Responsible use* section of the README.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
