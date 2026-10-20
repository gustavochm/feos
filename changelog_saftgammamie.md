# Changelog of the SAFT-γ Mie branch
Changes of the `saftgammamie` branch of the fork [gustavochm/feos](https://github.com/gustavochm/feos), kept
separate from FeOs' `CHANGELOG.md` so that rebases onto upstream never conflict. Temporary: the entries move to
`CHANGELOG.md` when the branch is proposed upstream.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- Added the read-only accessors `GcParameters::segment_records` and `GcParameters::binary_segment_records` (shared code in `feos-core`; needed by SAFT-γ Mie for unlike group pairs within a molecule, which `binary` and `association` omit).
- Added SAFT-γ Mie parameters (`saftgammamie` feature, in `all_models`): group, binary group and association records, `SaftGammaMieParameters`, `SaftGammaMiePars` (combining rules of Haslam et al. (2020) for all group pairs within and between molecules, molecular averages, input checks), hard-sphere diameter as in SAFT-VR Mie, and the model's association parameters including intramolecular overrides.
- Added the SAFT-γ Mie equation of state `SaftGammaMie` (residual Helmholtz energy with the contributions "Hard Sphere", "Dispersion", "Chain" and "Association"): monomer terms over all group pairs and the chain term from molecular averages (Haslam et al. (2020)), association strength Δ = F K I with the kernel of Dufal et al. (2015) and without hard-sphere contact value (override of `AssociationStrength::association_strength`). Tests: exact reduction to SAFT-VR Mie for single-group molecules, normal boiling points with published groups, subsets, merged groups, association kernel, dual-number derivatives.
