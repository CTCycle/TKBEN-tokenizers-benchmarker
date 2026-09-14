# Release Procedure
Last updated: 2026-09-14

## Release model

TKBEN publishes source-only GitHub Releases. The repository has no Tauri,
installer, executable, portable-app, or other binary packaging workflow. Do not
add packaging as part of a source release.

The public release version and component versions use the existing repository
convention:

| Surface | `v3.9.0` | `v4.0.0` | `v4.1.0` | `v4.2.0` | `v4.3.0` public |
| --- | ---: | ---: | ---: | ---: | ---: |
| Public Git tag and GitHub Release | `3.9.0` | `4.0.0` | `4.1.0` | `4.2.0` | `4.3.0` |
| Backend package (`app/server/pyproject.toml`) | `2.4.0` | `3.0.0` | `3.1.0` | `3.2.0` | `3.3.0` |
| Frontend package (`app/client/package.json`) | `1.4.0` | `2.0.0` | `2.1.0` | `2.2.0` | `2.3.0` |

The latest published release is `v4.3.0`. Its component versions are backend
`3.3.0` and frontend `2.3.0`.

## v4.3.0 release notes

The current release delta is based on verified repository changes after
`v4.2.0`:

- centralize configuration ownership between `.env` and structured JSON;
- load the environment before configuration and database imports;
- reject invalid boolean aliases, missing structured settings, and unknown
  configuration blocks;
- remove legacy configuration, namespace, cache, and tokenizer compatibility
  paths in favor of the canonical implementations;
- keep custom tokenizer identity and its canonical `tokenizer.json` artifact
  together across service recreation and persistence;
- make the Windows launcher build missing Angular production output before
  preview, verify the required `index.html`, and fail when an existing port
  listener cannot be stopped; redirected launches now capture backend logs for
  actionable health-check failures.

API version `1.2.0`, benchmark schema version `3`, report version `5`, and
Alembic revision `0003_canonical_state_cleanup` remain unchanged. Validation
evidence for this preparation is recorded under `assets/QA/`.

## Preparation and validation

1. Start from the current `develop`, inspect `git status`, the previous release
   tag, and the commits on `develop` since that tag. Preserve unrelated local
   work and do not make release-preparation edits directly on `main`.
2. Update the README, `assets/docs`, and this release procedure before the
   final branch synchronization. Keep documentation version references
   consistent with the release candidate.
3. Run the CI-equivalent checks for the intended release surfaces: backend
   compileall, Ruff, BasedPyright, unit tests, and OpenAPI smoke; frontend
   `npm run lint`, `npm run test:unit`, and `npm run build` from `app/client`.
4. Launch with `start_on_windows.ps1 -Launch`, verify the backend health
   endpoint and frontend production entry, then exercise Dataset, Tokenizers, and Cross
   Benchmark routes. Prioritize flows changed since the previous release,
   including current schema-3/report-5 persistence, catalog filtering, custom
   tokenizer handling, vocabulary-shape metrics, histogram/CDF views, bounded
   word-cloud layout, and dashboard visualization controls.
5. Run the focused live API/UI tests when the local services are available.
   Inspect browser console output and application logs; classify expected
   test-injected failures separately from release-blocking errors.
6. Record the validation evidence in `assets/QA/`. Do not claim skipped,
   unavailable-provider, or unverified checks as passed.

## Versioning and synchronization

After validation is release-ready, apply the coordinated minor bump to the
public tag version, backend package, frontend package, README, and relevant
documentation. For the current preparation, the public version is `v4.3.0`,
the backend package is `3.3.0`, and the frontend package is `2.3.0`. Commit all
release-preparation changes on `develop` before synchronizing branches.

Synchronize `main` from the validated `develop` commit so the branches point to
the same tree. Verify both branch tips and the clean worktree before creating
an annotated tag named `vX.Y.0` from the synchronized `main` commit.

## Publication

Create the GitHub Release from the annotated tag and use the release summary to
describe the reviewed delta and validation evidence. The source archive is the
only release artifact. After publication, verify that the tag and release point
to the synchronized `main` commit and that `develop` remains aligned with it.
