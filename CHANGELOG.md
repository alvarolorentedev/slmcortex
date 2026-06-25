# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project uses Semantic
Versioning for public releases.

## [0.1.0] - 2026-06-25

### Added

- Public Skill Cortex README with a single product-first narrative
- Architecture overview and support matrix for first-time users
- MIT license for the open-source v0.1 release
- Contributing guide, issue templates, and pull request template
- GitHub Actions workflows for pytest and no-model demo validation
- Dedicated v0.1.0 release notes under `docs/releases/`

### Changed

- Canonical public identity is now Skill Cortex in top-level docs and package
  metadata
- Legacy research workflow guidance moved out of the top-level README into
  dedicated docs
- Repo boundary documentation now distinguishes the public product surface from
  the legacy research surface

### Notes

- The legacy `skill-lattice` research CLI remains available for backward
  compatibility
- Product functionality, package contracts, runtime bundle contracts, adapters,
  and benchmarks are unchanged in this release
