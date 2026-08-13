# Release Procedure
Last updated: 2026-08-13

## Release model

TKBEN publishes source-only GitHub Releases. The repository has no Tauri,
installer, executable, portable-app, or other binary packaging workflow. Do not
add packaging as part of a source release.

The public release version and component versions use the existing repository
convention:

| Surface | `v3.8.0` | `v3.9.0` |
| --- | ---: | ---: |
| Public Git tag and GitHub Release | `3.8.0` | `3.9.0` |
| Backend package (`app/server/pyproject.toml`, `app/server/uv.lock`) | `2.3.0` | `2.4.0` |
| Frontend package and lockfile | `1.3.0` | `1.4.0` |

## Preparation and validation

1. Start from the latest `main` and inspect `git status`, the previous release
   tag, and the commits on `develop` since that tag. Preserve unrelated local
   work.
2. Update the README, `assets/docs`, and this release procedure before the
   final branch synchronization. Keep documentation version references
   consistent with the release being prepared.
3. Build the intended release surfaces: `npm --prefix app/client run lint`,
   `npm --prefix app/client run build`, and backend compile/test checks using
   `app/server/.venv` or the managed launcher environment.
4. Launch with `start_on_windows.ps1 -Launch`, verify the backend health
   endpoint and frontend, then exercise Dataset, Tokenizers, and Cross
   Benchmark routes. Prioritize flows changed since the previous release,
   including current schema-3/report-5 persistence, catalog filtering, custom
   tokenizer handling, and dashboard layout behavior.
5. Run the focused live API/UI tests when the local services are available.
   Inspect browser console output and application logs; classify expected
   test-injected failures separately from release-blocking errors.
6. Record the validation evidence in `assets/QA/`. Do not claim skipped,
   unavailable-provider, or unverified checks as passed.

## Versioning and synchronization

After validation is release-ready, apply the coordinated minor bump to the
public tag version, backend package and lockfile, frontend package and
lockfile, README, and relevant documentation. Commit all release-preparation
changes on `develop` before synchronizing branches.

Synchronize `main` from the validated `develop` commit so the branches point to
the same tree. Verify both branch tips and the clean worktree before creating
an annotated tag named `vX.Y.0` from the synchronized `main` commit.

## Publication

Create the GitHub Release from the annotated tag and use the release summary to
describe the reviewed delta and validation evidence. The source archive is the
only release artifact. After publication, verify that the tag and release point
to the synchronized `main` commit and that `develop` remains aligned with it.
