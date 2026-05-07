# Changelog

## [Unreleased]

### Notes

- No unreleased runtime changes are documented in this changelog.

## [v0.1.0] - 2026-05-07

### Added

- Documented the current SubspaceNetCC research codebase, including snapshot-domain data, covariance-domain processing, channel cross-correlation-domain features, and subspace-method evaluation paths.
- Documented the implemented `SubspaceNetCC` model path that consumes `[batch_size, G, 2K, N]` channel cross-correlation tensors and passes a surrogate covariance matrix to differentiable Root-MUSIC or ESPRIT.
- Documented the implemented CC-Toeplitz Root-MUSIC baseline built from segmented reference-channel cross-correlations, Hermitian Toeplitz reconstruction, and Root-MUSIC.
- Added per-version release notes for `v0.1.0`.

### Notes

- No experimental performance claims are included. The documentation is based on the current code, tests, and repository configuration.
- During this documentation update, the two CC check scripts could not run in the active Python environment because required packages were missing.
