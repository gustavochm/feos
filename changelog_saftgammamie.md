# Changelog of the SAFT-γ Mie branch
Changes of the `saftgammamie` branch of the fork [gustavochm/feos](https://github.com/gustavochm/feos), kept
separate from FeOs' `CHANGELOG.md` so that rebases onto upstream never conflict. Temporary: the entries move to
`CHANGELOG.md` when the branch is proposed upstream.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- Added the read-only accessors `GcParameters::segment_records` and `GcParameters::binary_segment_records` (shared code in `feos-core`; needed by SAFT-γ Mie for unlike group pairs within a molecule, which `binary` and `association` omit).
