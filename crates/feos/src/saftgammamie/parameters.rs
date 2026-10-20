use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::parameter::{
    AssociationParameters, BinaryParameters, BinarySegmentRecord, CombiningRule, GcParameters,
};
use feos_core::{FeosError, FeosResult};
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 10-point Gauss-Legendre quadrature [position, weight]
// copied from saftvrmie::parameters
const GLQ10: [[f64; 2]; 10] = [
    [-0.1488743389816312, 0.2955242247147529],
    [0.1488743389816312, 0.2955242247147529],
    [-0.4333953941292472, 0.2692667193099963],
    [0.4333953941292472, 0.2692667193099963],
    [-0.6794095682990244, 0.219086362515982],
    [0.6794095682990244, 0.219086362515982],
    [-0.8650633666889845, 0.1494513491505806],
    [0.8650633666889845, 0.1494513491505806],
    [-0.9739065285171717, 0.0666713443086881],
    [0.9739065285171717, 0.0666713443086881],
];

/// Admissible range of the Mie exponents, 5 < λ ≤ 100 (range of the correlations of Lafitte et al. (2013)).
const LAMBDA_MIN: f64 = 5.0;
const LAMBDA_MAX: f64 = 100.0;
/// Relative tolerance for exponents of corrected group pairs, which carry round-off from `gamma_kl`.
const LAMBDA_PAIR_RTOL: f64 = 1e-12;

/// SAFT-γ Mie group parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SaftGammaMieRecord {
    /// Number of identical segments of the group (ν*)
    pub vk: f64,
    /// Shape factor, the fraction of a segment that contributes (0 < S ≤ 1)
    pub shape_factor: f64,
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Repulsive Mie exponent
    pub lr: f64,
    /// Attractive Mie exponent
    pub la: f64,
}

impl SaftGammaMieRecord {
    pub fn new(vk: f64, shape_factor: f64, sigma: f64, epsilon_k: f64, lr: f64, la: f64) -> Self {
        Self {
            vk,
            shape_factor,
            sigma,
            epsilon_k,
            lr,
            la,
        }
    }
}

/// SAFT-γ Mie association parameters of a site pair.
#[derive(Serialize, Deserialize, Clone, Copy, Default, PartialEq, Debug)]
pub struct SaftGammaMieAssociationRecord {
    /// Bonding volume in units of Angstrom^3
    pub kappa_ab: f64,
    /// Association energy parameter in units of Kelvin
    pub epsilon_k_ab: f64,
}

impl SaftGammaMieAssociationRecord {
    pub fn new(kappa_ab: f64, epsilon_k_ab: f64) -> Self {
        Self {
            kappa_ab,
            epsilon_k_ab,
        }
    }
}

/// Combining rules of Haslam et al. (2020), Eq. 39-40: geometric mean of the association
/// energies, cube-root mean of the bonding volumes.
impl CombiningRule<SaftGammaMieRecord> for SaftGammaMieAssociationRecord {
    fn combining_rule(
        _: &SaftGammaMieRecord,
        _: &SaftGammaMieRecord,
        parameters_i: &Self,
        parameters_j: &Self,
    ) -> Self {
        let kappa_ab =
            (0.5 * (parameters_i.kappa_ab.cbrt() + parameters_j.kappa_ab.cbrt())).powi(3);
        let epsilon_k_ab = (parameters_i.epsilon_k_ab * parameters_j.epsilon_k_ab).sqrt();
        Self {
            kappa_ab,
            epsilon_k_ab,
        }
    }
}

/// SAFT-γ Mie binary group interaction parameters.
///
/// The corrections apply to any pair of different groups, within one molecule or between molecules.
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct SaftGammaMieBinaryRecord {
    /// Binary interaction parameter of the dispersion energy, ε_kl = (1 - k_kl) ε_kl^CR
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub k_kl: f64,
    /// Binary interaction parameter of the repulsive exponent, λ^r_kl = 3 + (1 - γ_kl) λ^r_kl^CR
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub gamma_kl: f64,
}

impl SaftGammaMieBinaryRecord {
    pub fn new(k_kl: Option<f64>, gamma_kl: Option<f64>) -> Self {
        let k_kl = k_kl.unwrap_or_default();
        let gamma_kl = gamma_kl.unwrap_or_default();
        Self { k_kl, gamma_kl }
    }
}

type GroupPairRecord = BinarySegmentRecord<SaftGammaMieBinaryRecord, SaftGammaMieAssociationRecord>;

/// Parameter set required for the SAFT-γ Mie equation of state.
pub type SaftGammaMieParameters<C> = GcParameters<
    SaftGammaMieRecord,
    SaftGammaMieBinaryRecord,
    SaftGammaMieAssociationRecord,
    (),
    C,
>;

/// The SAFT-γ Mie parameters in an easier accessible format.
///
/// Group quantities are indexed by the groups of all molecules (k, l), molecular averages of the
/// chain and association terms by the components (i, j). Binary group parameters are taken from
/// [GcParameters::binary_segment_records] for every pair of different groups, within and between
/// molecules; `parameters.binary` and `parameters.association`, which omit pairs within one molecule,
/// are not used by this model.
pub struct SaftGammaMiePars {
    /// Component (molecule) of each group
    pub component_index: DVector<usize>,
    /// Segment number of each group in its molecule, m_k,i = ν_k,i ν*_k S_k
    pub m: DVector<f64>,
    pub sigma: DVector<f64>,
    pub epsilon_k: DVector<f64>,
    pub lr: DVector<f64>,
    pub la: DVector<f64>,
    pub sigma_kl: DMatrix<f64>,
    pub epsilon_k_kl: DMatrix<f64>,
    pub lr_kl: DMatrix<f64>,
    pub la_kl: DMatrix<f64>,
    pub c_kl: DMatrix<f64>,
    pub alpha_kl: DMatrix<f64>,
    /// Segment number of each component, Σ_k m_k,i
    pub m_i: DVector<f64>,
    /// Molecular averages (Haslam et al. (2020), Eq. 14-18)
    pub sigma3_bar: DVector<f64>,
    pub sigma_bar: DVector<f64>,
    pub epsilon_k_bar: DVector<f64>,
    pub lr_bar: DVector<f64>,
    pub la_bar: DVector<f64>,
    pub c_bar: DVector<f64>,
    pub alpha_bar: DVector<f64>,
    /// Unlike molecular averages of the association kernel (Haslam et al. (2020), Eq. 26-27)
    pub sigma_bar_ij: DMatrix<f64>,
    pub epsilon_k_bar_ij: DMatrix<f64>,
}

impl SaftGammaMiePars {
    pub fn new(parameters: &SaftGammaMieParameters<f64>) -> FeosResult<Self> {
        let n = parameters.pure.len();
        let n_comp = parameters.molar_weight.len();
        let component_index: DVector<usize> = parameters.component_index().into();
        let identifiers: Vec<_> = parameters
            .pure
            .iter()
            .map(|g| g.identifier.as_str())
            .collect();

        for g in &parameters.pure {
            let r = &g.model_record;
            let id = &g.identifier;
            if !positive(r.vk) {
                return Err(incompatible(format!("vk of group {id} must be positive")));
            }
            if !(positive(r.shape_factor) && r.shape_factor <= 1.0) {
                return Err(incompatible(format!(
                    "shape factor of group {id} must satisfy 0 < S <= 1"
                )));
            }
            if !(positive(r.sigma) && positive(r.epsilon_k)) {
                return Err(incompatible(format!(
                    "sigma and epsilon_k of group {id} must be positive"
                )));
            }
            check_exponents(r.lr, r.la, 0.0, || format!("group {id}"))?;
        }

        let [vk, shape_factor] = parameters.collate(|pr| [pr.vk, pr.shape_factor]);
        let [sigma, epsilon_k] = parameters.collate(|pr| [pr.sigma, pr.epsilon_k]);
        let [lr, la] = parameters.collate(|pr| [pr.lr, pr.la]);
        let m = vk
            .component_mul(&shape_factor)
            .component_mul(&parameters.segment_counts());

        let binary = binary_records(parameters)?;
        let mut sigma_kl = DMatrix::zeros(n, n);
        let mut epsilon_k_kl = DMatrix::zeros(n, n);
        let mut lr_kl = DMatrix::zeros(n, n);
        let mut la_kl = DMatrix::zeros(n, n);
        let mut c_kl = DMatrix::zeros(n, n);
        let mut alpha_kl = DMatrix::zeros(n, n);
        for k in 0..n {
            for l in k..n {
                let (k_kl, gamma_kl) = binary
                    .get(&pair_key(identifiers[k], identifiers[l]))
                    .and_then(|br| br.model_record)
                    .map_or((0.0, 0.0), |br| (br.k_kl, br.gamma_kl));
                // combining rules as in saftvrmie::parameters (SaftVRMiePars::new)
                let s = 0.5 * (sigma[k] + sigma[l]);
                let e_k = (sigma[k].powi(3) * sigma[l].powi(3)).sqrt() / s.powi(3)
                    * (epsilon_k[k] * epsilon_k[l]).sqrt();
                let e = (1.0 - k_kl) * e_k;
                let r = (1.0 - gamma_kl) * ((lr[k] - 3.0) * (lr[l] - 3.0)).sqrt() + 3.0;
                let a = ((la[k] - 3.0) * (la[l] - 3.0)).sqrt() + 3.0;
                if k != l {
                    let what = || format!("group pair {}-{}", identifiers[k], identifiers[l]);
                    if !positive(e) {
                        return Err(incompatible(format!(
                            "epsilon_k of {} must be positive (k_kl < 1)",
                            what()
                        )));
                    }
                    check_exponents(r, a, LAMBDA_PAIR_RTOL, what)?;
                }
                let c = mie_prefactor(r, a);
                for (i, j) in [(k, l), (l, k)] {
                    sigma_kl[(i, j)] = s;
                    epsilon_k_kl[(i, j)] = e;
                    lr_kl[(i, j)] = r;
                    la_kl[(i, j)] = a;
                    c_kl[(i, j)] = c;
                    alpha_kl[(i, j)] = c * ((a - 3.0).recip() - (r - 3.0).recip());
                }
            }
        }

        // molecular averages with the weights z_k,i = m_k,i / m_i (Haslam et al. (2020), Eq. 14-18)
        let mut m_i: DVector<f64> = DVector::zeros(n_comp);
        for k in 0..n {
            m_i[component_index[k]] += m[k];
        }
        let mut sigma3_bar: DVector<f64> = DVector::zeros(n_comp);
        let mut epsilon_k_bar: DVector<f64> = DVector::zeros(n_comp);
        let mut lr_bar = DVector::zeros(n_comp);
        let mut la_bar = DVector::zeros(n_comp);
        for k in 0..n {
            for l in 0..n {
                let i = component_index[k];
                if component_index[l] != i {
                    continue;
                }
                let zz = m[k] * m[l] / (m_i[i] * m_i[i]);
                sigma3_bar[i] += zz * sigma_kl[(k, l)].powi(3);
                epsilon_k_bar[i] += zz * epsilon_k_kl[(k, l)];
                lr_bar[i] += zz * lr_kl[(k, l)];
                la_bar[i] += zz * la_kl[(k, l)];
            }
        }
        let sigma_bar = sigma3_bar.map(f64::cbrt);
        // C and α from the averaged exponents, not averages of C_kl (as sgtpy)
        let c_bar = lr_bar.zip_map(&la_bar, mie_prefactor);
        let alpha_bar = DVector::from_fn(n_comp, |i, _| {
            c_bar[i] * ((la_bar[i] - 3.0).recip() - (lr_bar[i] - 3.0).recip())
        });
        let sigma_bar_ij =
            DMatrix::from_fn(n_comp, n_comp, |i, j| 0.5 * (sigma_bar[i] + sigma_bar[j]));
        let epsilon_k_bar_ij = DMatrix::from_fn(n_comp, n_comp, |i, j| {
            (sigma3_bar[i] * sigma3_bar[j]).sqrt() / sigma_bar_ij[(i, j)].powi(3)
                * (epsilon_k_bar[i] * epsilon_k_bar[j]).sqrt()
        });

        Ok(Self {
            component_index,
            m,
            sigma,
            epsilon_k,
            lr,
            la,
            sigma_kl,
            epsilon_k_kl,
            lr_kl,
            la_kl,
            c_kl,
            alpha_kl,
            m_i,
            sigma3_bar,
            sigma_bar,
            epsilon_k_bar,
            lr_bar,
            la_bar,
            c_bar,
            alpha_bar,
            sigma_bar_ij,
            epsilon_k_bar_ij,
        })
    }
}

/// Strictly positive (false for NaN).
fn positive(x: f64) -> bool {
    x > 0.0
}

fn incompatible(message: String) -> FeosError {
    FeosError::IncompatibleParameters(message)
}

/// Checks 5 < λ^a < λ^r ≤ 100, with a relative tolerance at the bounds (for corrected pairs).
fn check_exponents<F: Fn() -> String>(lr: f64, la: f64, rtol: f64, what: F) -> FeosResult<()> {
    let in_range = |l: f64| l > LAMBDA_MIN * (1.0 - rtol) && l <= LAMBDA_MAX * (1.0 + rtol);
    if in_range(lr) && in_range(la) && lr > la {
        Ok(())
    } else {
        Err(incompatible(format!(
            "Mie exponents of {} must satisfy {LAMBDA_MIN} < la < lr <= {LAMBDA_MAX} (lr = {lr}, la = {la})",
            what()
        )))
    }
}

/// Prefactor of the Mie potential (Haslam et al. (2020), Eq. 2), as in saftvrmie::parameters.
fn mie_prefactor(lr: f64, la: f64) -> f64 {
    lr / (lr - la) * (lr / la).powf(la / (lr - la))
}

fn pair_key<'a>(id1: &'a str, id2: &'a str) -> (&'a str, &'a str) {
    if id1 <= id2 { (id1, id2) } else { (id2, id1) }
}

/// Binary segment records by (ordered) pair of group identifiers, for all group pairs.
///
/// A record of a group with itself and two records of the same pair are rejected; association
/// parameters of the records must not be negative.
fn binary_records(
    parameters: &SaftGammaMieParameters<f64>,
) -> FeosResult<HashMap<(&str, &str), &GroupPairRecord>> {
    let mut records = HashMap::new();
    for br in parameters.binary_segment_records().unwrap_or(&[]) {
        if br.id1 == br.id2 {
            return Err(incompatible(format!(
                "binary record of group {} with itself; the like pair has no corrections",
                br.id1
            )));
        }
        for a in &br.association_sites {
            check_association(&a.parameters, || {
                format!("binary record {}-{}", br.id1, br.id2)
            })?;
        }
        if records.insert(pair_key(&br.id1, &br.id2), br).is_some() {
            return Err(incompatible(format!(
                "more than one binary record for groups {} and {}",
                br.id1, br.id2
            )));
        }
    }
    Ok(records)
}

fn check_association<F: Fn() -> String>(
    record: &SaftGammaMieAssociationRecord,
    what: F,
) -> FeosResult<()> {
    if record.kappa_ab >= 0.0 && record.epsilon_k_ab >= 0.0 {
        Ok(())
    } else {
        Err(incompatible(format!(
            "kappa_ab and epsilon_k_ab of {} must not be negative",
            what()
        )))
    }
}

/// Association parameters of SAFT-γ Mie.
///
/// Binary association records apply to every pair of different groups, within and between molecules;
/// all other site pairs follow the combining rules. The association parameters stored in
/// `parameters.association`, which omit pairs within one molecule, are not used by this model.
pub fn association_parameters(
    parameters: &SaftGammaMieParameters<f64>,
) -> FeosResult<AssociationParameters<SaftGammaMieAssociationRecord>> {
    let segments: HashMap<_, _> = parameters
        .segment_records()
        .iter()
        .map(|s| (s.identifier.as_str(), s))
        .collect();
    let sites: Vec<_> = parameters
        .pure
        .iter()
        .map(|g| segments[g.identifier.as_str()].association_sites.clone())
        .collect();
    for (g, sites) in parameters.pure.iter().zip(&sites) {
        for site in sites.iter().filter_map(|s| s.parameters.as_ref()) {
            check_association(site, || format!("group {}", g.identifier))?;
        }
    }

    let binary = binary_records(parameters)?;
    let mut overrides = Vec::new();
    for (k, g1) in parameters.pure.iter().enumerate() {
        for (l, g2) in parameters.pure.iter().enumerate().skip(k + 1) {
            if let Some(br) = binary.get(&pair_key(&g1.identifier, &g2.identifier))
                && !br.association_sites.is_empty()
            {
                // the record lists sites of id1 first; orient it to the groups (k, l)
                let (k, l) = if br.id1 == g1.identifier {
                    (k, l)
                } else {
                    (l, k)
                };
                overrides.push(BinaryParameters::new(
                    k,
                    l,
                    br.association_sites.clone(),
                    (),
                ));
            }
        }
    }
    AssociationParameters::new_hetero(&parameters.pure, &sites, &overrides)
}

impl SaftGammaMiePars {
    /// Hard-sphere diameter of group k (Barker-Henderson), as `SaftVRMiePars::hs_diameter_ij`.
    #[inline]
    pub fn hs_diameter_kk<D: DualNum<f64> + Copy>(&self, k: usize, inverse_temperature: D) -> D {
        // copied from saftvrmie::parameters (hs_diameter_ij with i = j = k)
        let lr = self.lr_kl[(k, k)];
        let la = self.la_kl[(k, k)];
        let c_eps_t = inverse_temperature * self.c_kl[(k, k)] * self.epsilon_k_kl[(k, k)];

        // perform integration in reduced distances, then multiply sigma
        // r0 is dimensionless
        let r0 = lower_integratal_limit(la, lr, c_eps_t);
        let width = (-r0 + 1.0) * 0.5;
        GLQ10.iter().fold(r0, |d, &[x, w]| {
            let r = width * x + width + r0;
            let u = beta_u_mie(r, la, lr, 1.0, c_eps_t);
            let f_u = -(-u).exp_m1();
            d + width * f_u * w
        }) * self.sigma_kl[(k, k)]
    }
}

/// Find lower limit for integration of the temperature dependent diameter
///
/// Method of Aasen et al.
/// Using starting value proposed in Clapeyron.jl (Andrés Riedemann)
// copied from saftvrmie::parameters
fn lower_integratal_limit<D: DualNum<f64> + Copy>(la: f64, lr: f64, c_eps_t: D) -> D {
    // initial value from repulsive contribution
    let k = (-c_eps_t.recip() * f64::EPSILON.ln()).ln();
    let mut r = (-k / lr).exp();
    // Halley's method
    for _ in 1..5 {
        let [u, u_du, du_d2u] = mie_potential_halley(r, la, lr, c_eps_t);
        let dr = u_du / (-u_du / du_d2u * 0.5 + 1.0);
        if u.re() < 0.0 {
            return r;
        }
        r -= dr;
    }
    r
}

/// Calculate the fractions f / df and df / d2f used for Halley's method,
///
/// f is the function to find the root of.
/// Here, f = -beta u_mie(r) - ln(EPS)
// copied from saftvrmie::parameters
#[inline]
fn mie_potential_halley<D: DualNum<f64> + Copy>(r: D, la: f64, lr: f64, c_eps_t: D) -> [D; 3] {
    let ri = r.recip();
    let plr = ri.powf(lr);
    let pla = ri.powf(la);
    let u = plr - pla;
    let dplr = plr * (-lr) * ri;
    let dpla = pla * (-la) * ri;
    let du_dr = dplr - dpla;
    let d2u_dr2 = (dplr * (-lr - 1.0) - dpla * (-la - 1.0)) * ri;

    let f = -c_eps_t * u - f64::EPSILON.ln();
    let df = -c_eps_t * du_dr;
    let d2f = -c_eps_t * d2u_dr2;
    [f, f / df, df / d2f]
}

/// Dimensionless Mie potential (divided by kT)
// copied from saftvrmie::parameters
#[inline]
fn beta_u_mie<D: DualNum<f64> + Copy>(r: D, la: f64, lr: f64, sigma: f64, c_eps_t: D) -> D {
    let ri = r.recip() * sigma;
    (ri.powf(lr) - ri.powf(la)) * c_eps_t
}

impl HardSphereProperties for SaftGammaMiePars {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<'_, N> {
        let m = self.m.map(N::from);
        MonomerShape::Heterosegmented([m.clone(), m.clone(), m.clone(), m], &self.component_index)
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> DVector<D> {
        let t_inv = temperature.recip();
        DVector::from_fn(self.m.len(), |k, _| self.hs_diameter_kk(k, t_inv))
    }
}

/// Utilities for running tests
#[doc(hidden)]
pub mod test_utils {
    use super::*;
    use feos_core::parameter::{
        AssociationRecord, BinaryAssociationRecord, ChemicalRecord, Identifier, SegmentRecord,
    };

    pub type GroupRecord = SegmentRecord<SaftGammaMieRecord, SaftGammaMieAssociationRecord>;
    pub type BinaryGroupRecord = GroupPairRecord;

    /// Like association of a group: sites H (donor, A) and e1 (acceptor, B) with the like-pair
    /// parameters on both, so that the combining rule gives the like pair.
    fn sites(
        n_h: f64,
        n_e1: f64,
        kappa_ab: f64,
        epsilon_k_ab: f64,
    ) -> Vec<AssociationRecord<SaftGammaMieAssociationRecord>> {
        let p = Some(SaftGammaMieAssociationRecord::new(kappa_ab, epsilon_k_ab));
        vec![
            AssociationRecord::with_id("H".into(), p, n_h, 0.0, 0.0),
            AssociationRecord::with_id("e1".into(), p, 0.0, n_e1, 0.0),
        ]
    }

    /// Groups of the sgtpy database `saftgamma_database_v018.xlsx` (Haslam et al. (2020));
    /// sgtpy site H = donor (A), e1 = acceptor (B).
    pub fn groups() -> Vec<GroupRecord> {
        vec![
            GroupRecord::new(
                "CH3".into(),
                15.03422,
                SaftGammaMieRecord::new(1.0, 0.57255, 4.0772, 256.77, 15.05, 6.0),
            ),
            GroupRecord::new(
                "CH2".into(),
                14.02638,
                SaftGammaMieRecord::new(1.0, 0.22932, 4.8801, 473.39, 19.871, 6.0),
            ),
            GroupRecord::with_association(
                "CH2OH".into(),
                31.03322,
                SaftGammaMieRecord::new(2.0, 0.58538, 3.4054, 407.22, 22.699, 6.0),
                sites(1.0, 2.0, 62.309, 2097.9),
            ),
            GroupRecord::with_association(
                "H2O".into(),
                18.01468,
                SaftGammaMieRecord::new(1.0, 1.0, 3.0063, 266.68, 17.02, 6.0),
                sites(2.0, 2.0, 101.69, 1985.4),
            ),
        ]
    }

    /// Binary record for an unlike energy (and repulsive exponent) given directly, as in the sgtpy
    /// database; converted to k_kl, gamma_kl relative to the combining rules.
    pub fn binary_from_direct(
        group_k: &SaftGammaMieRecord,
        group_l: &SaftGammaMieRecord,
        epsilon_k_kl: f64,
        lr_kl: Option<f64>,
    ) -> SaftGammaMieBinaryRecord {
        let s = 0.5 * (group_k.sigma + group_l.sigma);
        let e_cr = (group_k.sigma.powi(3) * group_l.sigma.powi(3)).sqrt() / s.powi(3)
            * (group_k.epsilon_k * group_l.epsilon_k).sqrt();
        let lr_cr = ((group_k.lr - 3.0) * (group_l.lr - 3.0)).sqrt();
        SaftGammaMieBinaryRecord::new(
            Some(1.0 - epsilon_k_kl / e_cr),
            lr_kl.map(|lr| 1.0 - (lr - 3.0) / lr_cr),
        )
    }

    /// Unlike group parameters of the sgtpy database (first-order groups only).
    pub fn binary_groups() -> Vec<BinaryGroupRecord> {
        let g: HashMap<_, _> = groups()
            .into_iter()
            .map(|g| (g.identifier.clone(), g))
            .collect();
        let direct = |k: &str, l: &str, e: f64, lr: Option<f64>| {
            let record = binary_from_direct(&g[k].model_record, &g[l].model_record, e, lr);
            BinaryGroupRecord::new(k.into(), l.into(), Some(record))
        };
        let assoc = |id1: &str, id2: &str, kappa_ab, epsilon_k_ab| {
            BinaryAssociationRecord::with_id(
                id1.into(),
                id2.into(),
                SaftGammaMieAssociationRecord::new(kappa_ab, epsilon_k_ab),
            )
        };
        let mut water_alcohol = direct("CH2OH", "H2O", 353.37, None);
        // sgtpy: H2O(e1)-CH2OH(H) and H2O(H)-CH2OH(e1); sites listed as (CH2OH, H2O)
        water_alcohol.association_sites = vec![
            assoc("H", "e1", 425.0, 621.68),
            assoc("e1", "H", 147.4, 2153.2),
        ];
        vec![
            direct("CH3", "CH2", 350.77, None),
            direct("CH3", "CH2OH", 333.2, None),
            direct("CH2", "CH2OH", 423.17, None),
            direct("CH3", "H2O", 358.18, Some(100.0)),
            direct("CH2", "H2O", 423.63, Some(100.0)),
            water_alcohol,
        ]
    }

    pub fn molecule(name: &str) -> ChemicalRecord {
        let segments: &[&str] = match name {
            "hexane" => &["CH3", "CH2", "CH2", "CH2", "CH2", "CH3"],
            "ethanol" => &["CH3", "CH2OH"],
            "water" => &["H2O"],
            _ => panic!("unknown test molecule {name}"),
        };
        let identifier = Identifier::new(None, Some(name), None, None, None, None);
        ChemicalRecord::new(
            identifier,
            segments.iter().map(|&s| s.into()).collect(),
            None,
        )
    }

    /// Parameters of the given test molecules with all groups and binary records above.
    pub fn parameters(names: &[&str]) -> SaftGammaMieParameters<f64> {
        let molecules = names.iter().map(|&n| molecule(n)).collect();
        SaftGammaMieParameters::from_segments_hetero(molecules, &groups(), Some(&binary_groups()))
            .unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::test_utils::*;
    use super::*;
    use approx::assert_relative_eq;
    use feos_core::parameter::{BinaryAssociationRecord, ChemicalRecord, Identifier};

    fn pars(names: &[&str]) -> SaftGammaMiePars {
        SaftGammaMiePars::new(&parameters(names)).unwrap()
    }

    fn index(parameters: &SaftGammaMieParameters<f64>, component: usize, id: &str) -> usize {
        parameters
            .pure
            .iter()
            .position(|g| g.component_index == component && g.identifier == id)
            .unwrap()
    }

    #[cfg(feature = "saftvrmie")]
    #[test]
    fn hs_diameter_equals_saftvrmie() {
        use crate::saftvrmie::SaftVRMieRecord;
        use crate::saftvrmie::parameters::SaftVRMiePars;
        use crate::saftvrmie::test_utils::test_parameters;
        let mut names: Vec<_> = test_parameters().into_keys().collect();
        names.sort();
        for name in names {
            let vrmie = &test_parameters()[name];
            let r: &SaftVRMieRecord = &vrmie.pure[0].model_record;
            let group = GroupRecord::new(
                "G".into(),
                1.0,
                SaftGammaMieRecord::new(r.m, 1.0, r.sigma, r.epsilon_k, r.lr, r.la),
            );
            let molecule = ChemicalRecord::new(Identifier::default(), vec!["G".into()], None);
            let gammamie =
                SaftGammaMieParameters::from_segments_hetero(vec![molecule], &[group], None);
            let gammamie = SaftGammaMiePars::new(&gammamie.unwrap()).unwrap();
            let vrmie = SaftVRMiePars::new(vrmie);
            for temperature in [50.0, 150.0, 300.0, 600.0, 1000.0] {
                assert_eq!(
                    gammamie.hs_diameter(temperature)[0],
                    vrmie.hs_diameter(temperature)[0],
                    "{name} at {temperature} K"
                );
            }
            assert_eq!(gammamie.m[0], r.m);
            assert_eq!(gammamie.c_kl[(0, 0)], vrmie.c_ij[(0, 0)]);
            assert_eq!(gammamie.alpha_kl[(0, 0)], vrmie.alpha_ij[(0, 0)]);
        }
    }

    #[test]
    fn combining_rules() {
        let parameters = parameters(&["hexane", "ethanol", "water"]);
        let p = SaftGammaMiePars::new(&parameters).unwrap();
        let ch3 = index(&parameters, 0, "CH3");
        let ch2 = index(&parameters, 0, "CH2");
        let ch3_e = index(&parameters, 1, "CH3");
        let ch2oh = index(&parameters, 1, "CH2OH");
        let h2o = index(&parameters, 2, "H2O");

        // pairs with a record reproduce the direct values of the database
        assert_relative_eq!(p.epsilon_k_kl[(ch3, ch2)], 350.77, max_relative = 1e-14);
        assert_relative_eq!(p.epsilon_k_kl[(ch2oh, h2o)], 353.37, max_relative = 1e-14);
        assert_relative_eq!(p.lr_kl[(ch3, h2o)], 100.0, max_relative = 1e-14);
        // pairs without a record follow the combining rules, also for the same group in different molecules
        assert_eq!(p.epsilon_k_kl[(ch3, ch3_e)], p.epsilon_k_kl[(ch3, ch3)]);
        assert_relative_eq!(p.epsilon_k_kl[(ch3, ch3)], 256.77, max_relative = 1e-15);
        assert_relative_eq!(p.epsilon_k_kl[(ch2, ch2oh)], 423.17, max_relative = 1e-14);
        let lr_cr = ((15.05f64 - 3.0) * (19.871 - 3.0)).sqrt() + 3.0;
        assert_relative_eq!(p.lr_kl[(ch3, ch2)], lr_cr, max_relative = 1e-15);
        assert_relative_eq!(p.la_kl[(ch3, ch2)], 6.0, max_relative = 1e-15);
        let (lr, la) = (p.lr_kl[(ch3, ch2)], 6.0);
        let c = lr / (lr - la) * (lr / la).powf(la / (lr - la));
        assert_relative_eq!(p.c_kl[(ch3, ch2)], c, max_relative = 1e-15);
        assert_relative_eq!(
            p.alpha_kl[(ch3, ch2)],
            c * (1.0 / (la - 3.0) - 1.0 / (lr - 3.0)),
            max_relative = 1e-15
        );
        for m in [
            &p.sigma_kl,
            &p.epsilon_k_kl,
            &p.lr_kl,
            &p.la_kl,
            &p.c_kl,
            &p.alpha_kl,
        ] {
            assert_eq!(m, &m.transpose());
        }

        // without binary records every pair follows the combining rules (not zero)
        let plain = SaftGammaMieParameters::from_segments_hetero(
            vec![molecule("hexane"), molecule("water")],
            &groups(),
            None,
        )
        .unwrap();
        let q = SaftGammaMiePars::new(&plain).unwrap();
        let (ch3, h2o) = (index(&plain, 0, "CH3"), index(&plain, 1, "H2O"));
        let (s1, s2) = (4.0772f64, 3.0063f64);
        let s = 0.5 * (s1 + s2);
        let e = (s1.powi(3) * s2.powi(3)).sqrt() / s.powi(3) * (256.77f64 * 266.68).sqrt();
        let lr = ((15.05f64 - 3.0) * (17.02 - 3.0)).sqrt() + 3.0;
        assert_relative_eq!(q.sigma_kl[(ch3, h2o)], s, max_relative = 1e-15);
        assert_relative_eq!(q.epsilon_k_kl[(ch3, h2o)], e, max_relative = 1e-15);
        assert_relative_eq!(q.lr_kl[(ch3, h2o)], lr, max_relative = 1e-15);
        assert_relative_eq!(q.la_kl[(ch3, h2o)], 6.0, max_relative = 1e-15);
    }

    #[test]
    fn molecular_averages_hexane() {
        let parameters = parameters(&["hexane"]);
        let p = SaftGammaMiePars::new(&parameters).unwrap();
        let (ch3, ch2) = (index(&parameters, 0, "CH3"), index(&parameters, 0, "CH2"));
        let (m3, m2) = (2.0 * 0.57255, 4.0 * 0.22932);
        let m = m3 + m2;
        assert_relative_eq!(p.m_i[0], m, max_relative = 1e-15);
        let (z3, z2) = (m3 / m, m2 / m);
        let avg = |x: &DMatrix<f64>| {
            z3 * z3 * x[(ch3, ch3)] + 2.0 * z3 * z2 * x[(ch3, ch2)] + z2 * z2 * x[(ch2, ch2)]
        };
        let s3 = p.sigma_kl.map(|s| s.powi(3));
        assert_relative_eq!(p.sigma3_bar[0], avg(&s3), max_relative = 1e-14);
        assert_relative_eq!(
            p.epsilon_k_bar[0],
            avg(&p.epsilon_k_kl),
            max_relative = 1e-14
        );
        assert_relative_eq!(p.lr_bar[0], avg(&p.lr_kl), max_relative = 1e-14);
        assert_relative_eq!(p.la_bar[0], 6.0, max_relative = 1e-14);
        // the intramolecular CH3-CH2 energy (k_kl ≠ 0) enters the average
        assert_relative_eq!(p.epsilon_k_kl[(ch3, ch2)], 350.77, max_relative = 1e-14);
        let (lr, la) = (p.lr_bar[0], p.la_bar[0]);
        let c = lr / (lr - la) * (lr / la).powf(la / (lr - la));
        assert_relative_eq!(p.c_bar[0], c, max_relative = 1e-15);
        assert_relative_eq!(
            p.alpha_bar[0],
            c * (1.0 / (la - 3.0) - 1.0 / (lr - 3.0)),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            p.sigma_bar[0].powi(3),
            p.sigma3_bar[0],
            max_relative = 1e-15
        );
    }

    #[test]
    fn pair_selection_within_and_between_molecules() {
        // an intramolecular record changes the pure component ...
        let without =
            SaftGammaMieParameters::from_segments_hetero(vec![molecule("hexane")], &groups(), None)
                .unwrap();
        let without = SaftGammaMiePars::new(&without).unwrap();
        let with = pars(&["hexane"]);
        assert!(with.epsilon_k_bar[0] != without.epsilon_k_bar[0]);
        assert_eq!(with.epsilon_k_kl[(0, 0)], without.epsilon_k_kl[(0, 0)]);

        // ... an intermolecular one only the cross pairs: ethanol and water alone equal their subsets
        let mixture = parameters(&["ethanol", "water"]);
        for (component, name) in [(0, "ethanol"), (1, "water")] {
            let pure = pars(&[name]);
            let subset = SaftGammaMiePars::new(&mixture.subset(&[component])).unwrap();
            assert_eq!(pure.epsilon_k_kl, subset.epsilon_k_kl);
            assert_eq!(pure.lr_kl, subset.lr_kl);
            assert_eq!(pure.epsilon_k_bar, subset.epsilon_k_bar);
            assert_eq!(pure.m_i, subset.m_i);
        }
        let mix = SaftGammaMiePars::new(&mixture).unwrap();
        let (ch2oh, h2o) = (index(&mixture, 0, "CH2OH"), index(&mixture, 1, "H2O"));
        assert_relative_eq!(mix.epsilon_k_kl[(ch2oh, h2o)], 353.37, max_relative = 1e-14);
    }

    #[test]
    fn invalid_binary_records() {
        let mut binary = binary_groups();
        binary.push(BinaryGroupRecord::new(
            "CH2".into(),
            "CH2".into(),
            Some(SaftGammaMieBinaryRecord::new(Some(0.1), None)),
        ));
        let p = SaftGammaMieParameters::from_segments_hetero(
            vec![molecule("hexane")],
            &groups(),
            Some(&binary),
        )
        .unwrap();
        assert!(SaftGammaMiePars::new(&p).is_err());

        let mut binary = binary_groups();
        binary.push(BinaryGroupRecord::new(
            "CH2".into(),
            "CH3".into(),
            Some(SaftGammaMieBinaryRecord::new(Some(0.1), None)),
        ));
        let p = SaftGammaMieParameters::from_segments_hetero(
            vec![molecule("hexane")],
            &groups(),
            Some(&binary),
        )
        .unwrap();
        assert!(SaftGammaMiePars::new(&p).is_err());
        assert!(association_parameters(&p).is_err());
    }

    #[test]
    fn association_combining_rule_and_overrides() {
        let parameters = parameters(&["ethanol", "water"]);
        let a = association_parameters(&parameters).unwrap();
        let (ch2oh, h2o) = (index(&parameters, 0, "CH2OH"), index(&parameters, 1, "H2O"));
        let value = |donor: usize, acceptor: usize| {
            a.binary_ab
                .iter()
                .find(|bp| {
                    a.sites_a[bp.id1].assoc_comp == donor
                        && a.sites_b[bp.id2].assoc_comp == acceptor
                })
                .map(|bp| bp.model_record)
                .unwrap()
        };
        // overrides in both directions, with their own values
        assert_eq!(
            value(ch2oh, h2o),
            SaftGammaMieAssociationRecord::new(425.0, 621.68)
        );
        assert_eq!(
            value(h2o, ch2oh),
            SaftGammaMieAssociationRecord::new(147.4, 2153.2)
        );
        // like pairs follow the combining rule of the group's own values (cube-root mean of K)
        let like = value(h2o, h2o);
        assert_relative_eq!(like.kappa_ab, 101.69, max_relative = 1e-15);
        assert_eq!(like.epsilon_k_ab, 1985.4);
        // site numbers include the group count
        let n_b: f64 = a
            .sites_b
            .iter()
            .filter(|s| s.assoc_comp == ch2oh)
            .map(|s| s.n)
            .sum();
        assert_eq!(n_b, 2.0);

        // without the binary records, the cross pairs follow the combining rule
        let plain = SaftGammaMieParameters::from_segments_hetero(
            vec![molecule("ethanol"), molecule("water")],
            &groups(),
            None,
        )
        .unwrap();
        let a = association_parameters(&plain).unwrap();
        let cross = a
            .binary_ab
            .iter()
            .find(|bp| a.sites_a[bp.id1].assoc_comp == ch2oh && a.sites_b[bp.id2].assoc_comp == h2o)
            .unwrap()
            .model_record;
        let kappa = (0.5 * (62.309f64.cbrt() + 101.69f64.cbrt())).powi(3);
        assert_relative_eq!(cross.kappa_ab, kappa, max_relative = 1e-15);
        assert_relative_eq!(
            cross.epsilon_k_ab,
            (2097.9f64 * 1985.4).sqrt(),
            max_relative = 1e-15
        );
    }

    #[test]
    fn intramolecular_association_override() {
        // a hypothetical diol HO-CH2-CH2-OH with an override between its two (distinct) OH groups
        let mut groups = groups();
        let mut oh2 = groups[2].clone();
        oh2.identifier = "CH2OH_b".into();
        groups.push(oh2);
        let override_ = BinaryAssociationRecord::with_id(
            "H".into(),
            "e1".into(),
            SaftGammaMieAssociationRecord::new(10.0, 3000.0),
        );
        let binary = vec![BinaryGroupRecord::with_association(
            "CH2OH".into(),
            "CH2OH_b".into(),
            None,
            vec![override_],
        )];
        let diol = ChemicalRecord::new(
            Identifier::default(),
            vec!["CH2OH".into(), "CH2OH_b".into()],
            None,
        );
        let p = SaftGammaMieParameters::from_segments_hetero(vec![diol], &groups, Some(&binary))
            .unwrap();
        let a = association_parameters(&p).unwrap();
        let found = a.binary_ab.iter().any(|bp| {
            a.sites_a[bp.id1].assoc_comp == 0
                && a.sites_b[bp.id2].assoc_comp == 1
                && bp.model_record == SaftGammaMieAssociationRecord::new(10.0, 3000.0)
        });
        assert!(found);
        // FeOs' own association parameters omit the intramolecular override
        let override_ = SaftGammaMieAssociationRecord::new(10.0, 3000.0);
        assert!(
            !p.association
                .binary_ab
                .iter()
                .any(|bp| bp.model_record == override_)
        );
    }

    #[test]
    fn merged_equals_unmerged() {
        // hexane with six distinct groups of identical parameters (no corrections among equal groups)
        let mut groups = groups();
        let ids = ["CH3", "CH2", "CH2_b", "CH2_c", "CH2_d", "CH3_b"];
        for id in ["CH2_b", "CH2_c", "CH2_d", "CH3_b"] {
            let mut g = groups[if id.starts_with("CH3") { 0 } else { 1 }].clone();
            g.identifier = id.into();
            groups.push(g);
        }
        let ch3_ch2 = binary_from_direct(
            &groups[0].model_record,
            &groups[1].model_record,
            350.77,
            None,
        );
        let mut binary = Vec::new();
        for a in ids {
            for b in ids {
                if a < b && a.starts_with("CH3") != b.starts_with("CH3") {
                    binary.push(BinaryGroupRecord::new(a.into(), b.into(), Some(ch3_ch2)));
                }
            }
        }
        let unmerged = ChemicalRecord::new(
            Identifier::default(),
            ids.iter().map(|&s| s.into()).collect(),
            None,
        );
        let unmerged =
            SaftGammaMieParameters::from_segments_hetero(vec![unmerged], &groups, Some(&binary))
                .unwrap();
        let unmerged = SaftGammaMiePars::new(&unmerged).unwrap();
        let merged = pars(&["hexane"]);
        assert_eq!(unmerged.m.len(), 6);
        assert_eq!(merged.m.len(), 2);
        for (u, m) in [
            (&unmerged.m_i, &merged.m_i),
            (&unmerged.sigma3_bar, &merged.sigma3_bar),
            (&unmerged.epsilon_k_bar, &merged.epsilon_k_bar),
            (&unmerged.lr_bar, &merged.lr_bar),
            (&unmerged.la_bar, &merged.la_bar),
            (&unmerged.c_bar, &merged.c_bar),
            (&unmerged.alpha_bar, &merged.alpha_bar),
        ] {
            assert_relative_eq!(u[0], m[0], max_relative = 1e-14);
        }
        // segment-weighted pair sum, as in the monomer terms
        let pair_sum = |p: &SaftGammaMiePars| {
            let n = p.m.len();
            (0..n)
                .flat_map(|k| (0..n).map(move |l| (k, l)))
                .map(|(k, l)| p.m[k] * p.m[l] * p.epsilon_k_kl[(k, l)])
                .sum::<f64>()
        };
        assert_relative_eq!(pair_sum(&unmerged), pair_sum(&merged), max_relative = 1e-14);
    }

    #[test]
    fn json_round_trip() {
        let groups_json = serde_json::to_string(&groups()).unwrap();
        let binary_json = serde_json::to_string(&binary_groups()).unwrap();
        let groups_back: Vec<GroupRecord> = serde_json::from_str(&groups_json).unwrap();
        let binary_back: Vec<BinaryGroupRecord> = serde_json::from_str(&binary_json).unwrap();
        // serde_json parses floats without `float_roundtrip`: equal up to one ulp
        let close = |a: &str, b: &str| {
            let numbers = |s: &str| -> Vec<f64> {
                s.split(|c: char| !(c.is_ascii_digit() || "-.e".contains(c)))
                    .filter_map(|t| t.parse().ok())
                    .collect()
            };
            let (a, b) = (numbers(a), numbers(b));
            assert_eq!(a.len(), b.len());
            a.iter()
                .zip(&b)
                .for_each(|(x, y)| assert_relative_eq!(x, y, max_relative = 1e-15));
        };
        close(&serde_json::to_string(&groups_back).unwrap(), &groups_json);
        close(&serde_json::to_string(&binary_back).unwrap(), &binary_json);
        let keys = |s: &str| {
            s.split('"')
                .skip(1)
                .step_by(2)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            keys(&serde_json::to_string(&binary_back).unwrap()),
            keys(&binary_json)
        );
        // file layout of the plan (MASTERPLAN_GAMMAMIE 4.4): field names as written by users
        let segment: GroupRecord = serde_json::from_str(
            r#"{"identifier": "CH2OH", "molarweight": 31.03, "vk": 2, "shape_factor": 0.58538,
                "sigma": 3.4054, "epsilon_k": 407.22, "lr": 22.699, "la": 6.0,
                "association_sites": [{"id": "s", "na": 1, "nb": 2, "kappa_ab": 62.309, "epsilon_k_ab": 2097.9}]}"#,
        )
        .unwrap();
        assert_eq!(segment.model_record.vk, 2.0);
        let binary: BinaryGroupRecord =
            serde_json::from_str(r#"{"id1": "CH3", "id2": "CH2", "k_kl": 0.02}"#).unwrap();
        assert_eq!(binary.model_record.unwrap().k_kl, 0.02);
        assert_eq!(binary.model_record.unwrap().gamma_kl, 0.0);
    }

    #[test]
    fn invalid_parameters() {
        let invalid = |f: fn(&mut SaftGammaMieRecord)| {
            let mut groups = groups();
            f(&mut groups[3].model_record);
            let p = SaftGammaMieParameters::from_segments_hetero(
                vec![molecule("water")],
                &groups,
                None,
            )
            .unwrap();
            SaftGammaMiePars::new(&p).is_err()
        };
        assert!(invalid(|r| r.vk = 0.0));
        assert!(invalid(|r| r.shape_factor = 0.0));
        assert!(invalid(|r| r.shape_factor = 1.1));
        assert!(invalid(|r| r.sigma = -1.0));
        assert!(invalid(|r| r.epsilon_k = 0.0));
        assert!(invalid(|r| r.la = 5.0));
        assert!(invalid(|r| r.lr = 100.1));
        assert!(invalid(|r| r.lr = 6.0));
        assert!(!invalid(|r| r.lr = 100.0));

        // corrected pairs: k_kl >= 1 and an exponent pushed below la
        let pair = |record: SaftGammaMieBinaryRecord| {
            let binary = [BinaryGroupRecord::new(
                "CH3".into(),
                "CH2".into(),
                Some(record),
            )];
            let p = SaftGammaMieParameters::from_segments_hetero(
                vec![molecule("hexane")],
                &groups(),
                Some(&binary),
            )
            .unwrap();
            SaftGammaMiePars::new(&p).is_err()
        };
        assert!(pair(SaftGammaMieBinaryRecord::new(Some(1.0), None)));
        assert!(pair(SaftGammaMieBinaryRecord::new(None, Some(0.9))));
        assert!(!pair(SaftGammaMieBinaryRecord::new(Some(0.1), Some(0.1))));

        // negative association parameters
        let mut groups = groups();
        groups[3].association_sites[0].parameters =
            Some(SaftGammaMieAssociationRecord::new(-1.0, 1985.4));
        let p =
            SaftGammaMieParameters::from_segments_hetero(vec![molecule("water")], &groups, None)
                .unwrap();
        assert!(association_parameters(&p).is_err());
    }
}
