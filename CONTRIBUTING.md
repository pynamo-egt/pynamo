# Contributing to pyNamo-EGT

Thanks for your interest in pyNamo-EGT! Bug reports, feature ideas, documentation fixes and code are all welcome.

## Asking questions and reporting problems

- **Questions** ("how do I…?"): see [SUPPORT.md](SUPPORT.md).
- **Bugs**: open a [bug report](https://github.com/pynamo-egt/pynamo/issues/new?template=bug_report.yml) with a minimal example, what you expected and what happened.
- **Ideas**: open a [feature request](https://github.com/pynamo-egt/pynamo/issues/new?template=feature_request.yml). For larger changes, please discuss in an issue before writing code.

## Development setup

pyNamo-EGT requires Python 3.12 or newer.

```bash
git clone https://github.com/pynamo-egt/pynamo.git
cd pynamo
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,plotly]"
python -m pytest -q
```

Add the `notebook` extra (`pip install -e ".[dev,plotly,notebook]"`) to work on the Jupyter widget or the tutorials.

## Making a change

1. Branch from `master`.
2. Keep the change small and focused: one issue per pull request, ideally under ~400 changed lines.
3. Add or update tests in the same pull request. Good tests check behaviour through the public API (analysis results, figure contents), are deterministic (seed any randomness) and are named after the behaviour they check.
4. Keep existing behaviour working. Code written for earlier 0.x releases must keep running; if your change alters an existing output, say so in the pull request description and cover it with a test. Deprecated behaviour keeps working, with a `DeprecationWarning`, until the next major release and for at least six months.
5. Run `python -m pytest -q` before pushing. CI runs the suite on Linux, macOS and Windows for every supported Python version, plus once on the oldest supported dependency versions.

## Pull request titles

Pull requests are squash-merged, and the **PR title becomes the commit message** on `master`. Releases are computed from these messages, so titles must follow [Conventional Commits](https://www.conventionalcommits.org/); a CI check enforces this.

```text
<type>(<optional scope>): <short summary in the imperative>
```

| Type | Use for | Release |
| --- | --- | --- |
| `feat` | a new feature | minor |
| `fix` | a bug fix | patch |
| `perf` | a performance improvement | patch |
| `docs`, `test`, `ci`, `build`, `chore`, `refactor`, `style`, `revert` | everything else | none |

Examples: `feat(analysis): add a basis column to the equilibrium table`, `fix(game): reject payoff tensors of the wrong rank`. Add `!` after the type (`feat!: …`) for a breaking change.

## Stacked pull requests

When a change depends on another pull request that is not merged yet:

- branch from the parent pull request's branch and set your PR's base to it;
- keep stacks short (at most three);
- they are merged bottom-up. After the bottom PR is squash-merged, rebase the next branch onto `master`, using the old parent tip as the upstream, and force-push with lease:

  ```bash
  git rebase --onto master <old-parent-tip> <your-branch>
  git push --force-with-lease
  ```

Independent changes should branch from `master`.

## Releases

Maintainers don't release by hand. Every merged `feat`, `fix` or `perf` pull request triggers a release: the version number, the changelog, the Git tag, the GitHub release and the PyPI upload are all produced automatically from the conventional PR titles.
