use super::PyEquationOfState;
use crate::error::PyFeosError;
use crate::{ideal_gas::IdealGasModel, parameter::PyGcParameters, residual::ResidualModel};
use feos::saftgammamie::{SaftGammaMie, SaftGammaMieOptions};
use feos_core::{EquationOfState, ResidualDyn};
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl PyEquationOfState {
    /// SAFT-γ Mie group contribution equation of state.
    ///
    /// Haslam et al., J. Chem. Eng. Data 65 (2020) 5862. Binary group records
    /// (`k_kl`, `gamma_kl` and association overrides) apply to every pair of
    /// different groups, within one molecule and between molecules.
    ///
    /// Parameters
    /// ----------
    /// parameters : GcParameters
    ///     The group parameters (`vk`, `shape_factor`, `sigma`, `epsilon_k`, `lr`, `la`,
    ///     association sites with `kappa_ab` in Angstrom^3 and `epsilon_k_ab`).
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The SAFT-γ Mie equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10),
        text_signature = "(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10)"
    )]
    pub fn saftgammamie(
        parameters: PyGcParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
    ) -> PyResult<Self> {
        let options = SaftGammaMieOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
        };
        let residual = ResidualModel::SaftGammaMie(
            SaftGammaMie::with_options(parameters.try_convert_heterosegmented()?, options)
                .map_err(PyFeosError::from)?,
        );
        let ideal_gas = vec![IdealGasModel::NoModel; residual.components()];
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}
