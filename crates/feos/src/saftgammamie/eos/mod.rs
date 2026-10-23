use super::parameters::{
    SaftGammaMieAssociationRecord, SaftGammaMieParameters, SaftGammaMiePars, association_parameters,
};
use crate::association::Association;
use crate::hard_sphere::HardSphere;
use feos_core::parameter::AssociationParameters;
use feos_core::{FeosResult, Molarweight, ResidualDyn, StateHD, Subset};
use nalgebra::DVector;
use num_dual::DualNum;
use quantity::MolarWeight;
use std::f64::consts::FRAC_PI_6;

pub(crate) mod association;
pub(crate) mod dispersion_chain;
pub(crate) mod temperature;
use dispersion_chain::{Properties, helmholtz_energy_density_disp_chain};

/// Customization options for the SAFT-γ Mie equation of state.
#[derive(Copy, Clone)]
pub struct SaftGammaMieOptions {
    pub max_eta: f64,
    pub max_iter_cross_assoc: usize,
    pub tol_cross_assoc: f64,
}

impl Default for SaftGammaMieOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
        }
    }
}

/// SAFT-γ Mie equation of state.
pub struct SaftGammaMie {
    pub parameters: SaftGammaMieParameters<f64>,
    pub params: SaftGammaMiePars,
    /// Association parameters of all group pairs, within and between molecules
    pub association_parameters: AssociationParameters<SaftGammaMieAssociationRecord>,
    pub options: SaftGammaMieOptions,
    /// any molecule with Σ_k m_k,i ≠ 1
    pub chain: bool,
    pub association: Option<Association>,
}

impl SaftGammaMie {
    pub fn new(parameters: SaftGammaMieParameters<f64>) -> FeosResult<Self> {
        Self::with_options(parameters, SaftGammaMieOptions::default())
    }

    pub fn with_options(
        parameters: SaftGammaMieParameters<f64>,
        options: SaftGammaMieOptions,
    ) -> FeosResult<Self> {
        let params = SaftGammaMiePars::new(&parameters)?;
        let association_parameters = association_parameters(&parameters)?;
        let chain = params.m_i.iter().any(|&m| m != 1.0);
        let association = (!association_parameters.is_empty())
            .then(|| Association::new(options.max_iter_cross_assoc, options.tol_cross_assoc));
        Ok(Self {
            parameters,
            params,
            association_parameters,
            options,
            chain,
            association,
        })
    }
}

impl ResidualDyn for SaftGammaMie {
    fn components(&self) -> usize {
        self.parameters.molar_weight.len()
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let p = &self.params;
        let molefracs_segments = p.component_index.map(|i| molefracs[i]);
        (p.m.component_mul(&p.sigma.map(|v| v.powi(3)))
            .map(D::from)
            .dot(&molefracs_segments)
            * FRAC_PI_6)
            .recip()
            * self.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        let mut a = Vec::with_capacity(4);

        let (a_hs, _, d) = HardSphere.helmholtz_energy_density_and_properties(&self.params, state);
        a.push(("Hard Sphere", a_hs));

        let properties = Properties::new(&self.params, state, &d);
        let (a_disp, a_chain) =
            helmholtz_energy_density_disp_chain(&self.params, &properties, state);
        a.push(("Dispersion", a_disp));
        if self.chain {
            a.push(("Chain", a_chain));
        }
        if let Some(assoc) = self.association.as_ref() {
            a.push((
                "Association",
                assoc.helmholtz_energy_density(
                    &self.params,
                    &self.association_parameters,
                    state,
                    &d,
                ),
            ));
        }
        a
    }
}

impl Subset for SaftGammaMie {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options).unwrap()
    }
}

impl Molarweight for SaftGammaMie {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}
