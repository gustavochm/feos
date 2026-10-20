# Changelog of the SAFT-γ Mie branch
Changes of the `saftgammamie` branch of the fork [gustavochm/feos](https://github.com/gustavochm/feos), kept
separate from FeOs' `CHANGELOG.md` so that rebases onto upstream never conflict. Temporary: the entries move to
`CHANGELOG.md` when the branch is proposed upstream.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Every entry names the git tag of the
task that introduced it (`saftgammamie-G<n>`, on the task's last commit; `git show saftgammamie-G2`), plus the commits.

## [Unreleased]
### Added
- [`saftgammamie-G1`, `54c3c34`] Added the read-only accessors `GcParameters::segment_records` and `GcParameters::binary_segment_records` (shared code in `feos-core`; needed by SAFT-γ Mie for unlike group pairs within a molecule, which `binary` and `association` omit).
- [`saftgammamie-G2`, `bb2b50d` (this changelog), `6ac6c50`] Added SAFT-γ Mie parameters (`saftgammamie` feature, in `all_models`): group, binary group and association records, `SaftGammaMieParameters`, `SaftGammaMiePars` (combining rules of Haslam et al. (2020) for all group pairs within and between molecules, molecular averages, input checks), hard-sphere diameter as in SAFT-VR Mie, and the model's association parameters including intramolecular overrides.
- [`saftgammamie-G3`, `c7c7d6c`] Added the SAFT-γ Mie equation of state `SaftGammaMie` (residual Helmholtz energy with the contributions "Hard Sphere", "Dispersion", "Chain" and "Association"): monomer terms over all group pairs and the chain term from molecular averages (Haslam et al. (2020)), association strength Δ = F K I with the kernel of Dufal et al. (2015) and without hard-sphere contact value (override of `AssociationStrength::association_strength`). Tests: exact reduction to SAFT-VR Mie for single-group molecules, normal boiling points with published groups, subsets, merged groups, association kernel, dual-number derivatives.
- [`saftgammamie-G4`] Added the Python constructor `EquationOfState.saftgammamie(parameters, max_eta, max_iter_cross_assoc, tol_cross_assoc)` (shared py-feos code: feature `saftgammamie`, `ResidualModel::SaftGammaMie`, `eos` module registration, and the gate of `PyGcParameters::try_convert_heterosegmented` widened to `any(gc_pcsaft, saftgammamie)`). Python tests live in gnnpcsaft (`tests/feos_io/test_gammamie_binding.py`).
- [`saftgammamie-G3b`] Faster SAFT-γ Mie (12-53 %, 2.1x for 40 distinct groups): powers of x0 shared within a group pair (two `powf` instead of fifteen; the only change to the kernels copied from SAFT-VR Mie), chain loops over the groups of each molecule, σ³_kl precomputed, association kernel once per molecule pair. Added the bench `saftgammamie` (`crates/feos/benches/saftgammamie.rs`).
