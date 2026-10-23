//! Monomer (dispersion) and chain contributions of SAFT-γ Mie.
//!
//! The structure and the kernels (`zeta_eff`, `a1s_ij`, `b_ij`, `a1s_b_ij`, `f`, `PHI`, `C`) are
//! copied from `saftvrmie::eos::dispersion`, with the powers of x0 shared within a pair (see
//! `a1s_b_terms`); sums run over the groups of all molecules (monomer) and over the molecular
//! averages (chain), Haslam et al. (2020), Eq. 6-20.
//!
//! Intended change to the copied kernels: the temperature-only factors of the monomer pairs (d_kl³,
//! x0^λ and the I_λ, J_λ of the B term) come from [TemperatureTerms], computed once per temperature,
//! so x0^λ (a1s + B) = x0^λ a1s + x0^λ I_λ g_I(ζ_x) − x0^λ J_λ g_J(ζ_x). The copied kernels remain for
//! dual numbers of higher order than three.
use super::temperature::{Delta, TemperatureTerms};
use crate::saftgammamie::parameters::SaftGammaMiePars;
use feos_core::StateHD;
use nalgebra::{DMatrix, DVector};
use num_dual::{Dual, DualNum};
use num_traits::Zero;
use std::f64::consts::{FRAC_PI_6, PI};

#[derive(Debug)]
pub struct Properties<D> {
    /// Temperature dependent diameter of each group
    diameter: DVector<D>,
    /// total number density of segments
    pub segment_density: D,
    /// segment fraction of each group, x_s,k (Eq. 11)
    pub segment_molefracs: DVector<D>,
    /// mean segment number
    mean_segment_number: D,
    /// mixture packing fraction using d(T)
    zeta_x: D,
    /// mixture packing fraction using sigma
    pub zeta_x_bar: D,
    /// k-values for HS pair correlation fn
    k0: [D; 4],
}

impl<D: DualNum<f64> + Copy + Zero> Properties<D> {
    pub(super) fn new(
        parameters: &SaftGammaMiePars,
        state: &StateHD<D>,
        diameter: &DVector<D>,
    ) -> Self {
        let n = parameters.m.len();
        // mole fraction of the molecule owning each group
        let x = parameters.component_index.map(|i| state.molefracs[i]);
        let m = parameters.m.map(D::from);

        let mean_segment_number: D = x.component_mul(&m).sum();
        let xs = x.component_mul(&m) / mean_segment_number;

        // Set eps to one -> get partial derivatives w.r.t segment density
        let segment_density = parameters
            .component_index
            .map(|i| state.partial_density[i])
            .component_mul(&m)
            .sum();

        // d_kl³ once per temperature (temperature.rs); directly for higher-order dual numbers
        let d3_kl = if D::NDERIV <= 3 {
            let terms = TemperatureTerms::get(parameters, state.temperature.re());
            let delta = Delta::new(state.temperature);
            DMatrix::from_fn(n, n, |k, l| delta.eval(&terms.pair(k, l).d3))
        } else {
            DMatrix::from_fn(n, n, |k, l| ((diameter[k] + diameter[l]) * 0.5).powi(3))
        };

        // segment packing fraction
        let mut zeta_x = D::zero();
        let mut zeta_x_bar = D::zero();
        for k in 0..n {
            zeta_x += xs[k].powi(2) * d3_kl[(k, k)];
            zeta_x_bar += xs[k].powi(2) * parameters.sigma3_kl[(k, k)];
            for l in k + 1..n {
                zeta_x += xs[k] * xs[l] * d3_kl[(k, l)] * 2.0;
                zeta_x_bar += xs[k] * xs[l] * parameters.sigma3_kl[(k, l)] * 2.0;
            }
        }
        zeta_x *= segment_density * FRAC_PI_6;
        zeta_x_bar *= segment_density * FRAC_PI_6;

        let frac_1mzeta3 = (-zeta_x + 1.0).powi(3).recip();
        let z = zeta_x;
        let z2 = z * z;
        let z3 = z2 * z;
        let k0 =
            -(-z + 1.0).ln() + z * (-z * 39.0 + z2 * 9.0 - z3 * 2.0 + 42.0) * frac_1mzeta3 / 6.0;
        let k1 = z * frac_1mzeta3 * 0.5 * (z3 + z * 6.0 - 12.0);
        let k2 = -z2 * 3.0 / 8.0 * frac_1mzeta3 * (-zeta_x + 1.0);
        let k3 = z * frac_1mzeta3 / 6.0 * (-z3 + z * 3.0 + 3.0);

        Self {
            diameter: diameter.clone(),
            segment_density,
            segment_molefracs: xs,
            mean_segment_number,
            zeta_x,
            zeta_x_bar,
            k0: [k0, k1, k2, k3],
        }
    }
}

pub(super) const PHI: [[f64; 7]; 6] = [
    [
        7.5365557, -37.60463, 71.745953, -46.83552, -2.467982, -0.50272, 8.0956883,
    ],
    [-359.44, 1825.6, -3168.0, 1884.2, -0.82376, -3.1935, 3.709],
    [1550.9, -5070.1, 6534.6, -3288.7, -2.7171, 2.0883, 0.0],
    [
        -1.19932, 9.063632, -17.9482, 11.34027, 20.52142, -56.6377, 40.53683,
    ],
    [
        -1911.28, 21390.175, -51320.7, 37064.54, 1103.742, -3264.61, 2556.181,
    ],
    [
        9236.9, -129430.0, 357230.0, -315530.0, 1390.2, -4518.2, 4241.6,
    ],
];

/// Monomer (dispersion) and chain contributions to the reduced Helmholtz energy density.
///
/// The monomer terms a1, a2, a3 run over all pairs of groups (Eq. 13); the chain term uses the
/// molecular averages of each molecule i with Σ_k m_k,i ≠ 1 (Eq. 19-20), whose derivatives of a1
/// and a2 with respect to the segment density are obtained with ρ_s as a dual number, as in
/// SAFT-VR Mie. Returns `(a_dispersion, a_chain)`.
pub fn helmholtz_energy_density_disp_chain<D: DualNum<f64> + Copy>(
    parameters: &SaftGammaMiePars,
    properties: &Properties<D>,
    state: &StateHD<D>,
) -> (D, D) {
    let p = &parameters;
    let n = p.sigma.len();
    let k = &properties.k0;
    let xs = &properties.segment_molefracs;
    let t_inv = state.temperature.inv();
    let zeta_x = properties.zeta_x;
    let k_hs = (zeta_x - 1.0).powi(4)
        / ((zeta_x + zeta_x.powi(2) - zeta_x.powi(3)) * 4.0 + zeta_x.powi(4) + 1.0);
    let zeta_x_bar = properties.zeta_x_bar;
    let zx5 = zeta_x_bar.powi(5);
    let zx8 = zeta_x_bar.powi(8);

    let mut a1 = D::zero();
    let mut a2 = D::zero();
    let mut a3 = D::zero();

    // temperature-only pair factors once per temperature (temperature.rs); directly for
    // higher-order dual numbers
    let cached = (D::NDERIV <= 3).then(|| {
        let terms = TemperatureTerms::get(p, state.temperature.re());
        (terms, Delta::new(state.temperature))
    });
    // ζ_x factors of the B term: g_I = (1 − ζ/2)/(1 − ζ)³, g_J = 9ζ(1 + ζ)/(2(1 − ζ)³)
    let frac_1mzeta3 = (-zeta_x + 1.0).powi(-3);
    let g_i = (-zeta_x * 0.5 + 1.0) * frac_1mzeta3;
    let g_j = zeta_x * (zeta_x + 1.0) * 4.5 * frac_1mzeta3;
    // d_kl³ and x0^λ (a1s + B) of the five exponents of the pair (k, l)
    let pair = |k: usize, l: usize, d_kl: D, sigma: f64, la: f64, lr: f64| -> (D, [D; 5]) {
        match &cached {
            Some((terms, delta)) => {
                let t = terms.pair(k, l);
                let lambdas = [la, lr, 2.0 * la, la + lr, 2.0 * lr];
                // x0^λ B = x0^λ I_λ g_I − x0^λ J_λ g_J (Lafitte et al. (2013), Eq. A12), as b_ij
                let a1s_b = std::array::from_fn(|m| {
                    delta.eval(&t.x0_lambda[m]) * a1s_ij(zeta_x, lambdas[m])
                        + delta.eval(&t.x0_lambda_i[m]) * g_i
                        - delta.eval(&t.x0_lambda_j[m]) * g_j
                });
                (delta.eval(&t.d3), a1s_b)
            }
            None => (
                d_kl.powi(3),
                a1s_b_terms(zeta_x, d_kl.recip() * sigma, la, lr),
            ),
        }
    };

    for i in 0..n {
        // parameters (like pair of group i, record values as in SAFT-VR Mie)
        let eps_k = p.epsilon_k[i];
        let sig = p.sigma[i];
        let la = p.la[i];
        let lr = p.lr[i];
        let c = p.c_kl[(i, i)];

        let di = properties.diameter[i];
        let (d3, [a1s_b_la, a1s_b_lr, a1s_b_2la, a1s_b_lalr, a1s_b_2lr]) =
            pair(i, i, di, sig, la, lr);
        let pref = properties.segment_density * d3 * eps_k * 2.0 * PI * c;
        let a1_ii = pref * (a1s_b_la - a1s_b_lr);
        let a2_ii = pref * eps_k * c * k_hs * 0.5 * (a1s_b_2la - a1s_b_lalr * 2.0 + a1s_b_2lr);

        // note indices of f(i, alpha) are shifted due to 0-indexing.
        let alpha = parameters.alpha_kl[(i, i)];
        let a3_ii = -zeta_x_bar
            * f(3, alpha)
            * (zeta_x_bar * (zeta_x_bar * f(5, alpha) + f(4, alpha))).exp()
            * eps_k.powi(3);

        let xii = zeta_x_bar * f(0, alpha) + zx5 * f(1, alpha) + zx8 * f(2, alpha);

        // accumulate contributions
        let xs_ii2 = xs[i] * xs[i];
        a1 += a1_ii * xs_ii2;
        a2 += a2_ii * xs_ii2 * (xii + 1.0);
        a3 += a3_ii * xs_ii2;

        for j in i + 1..n {
            // parameters (unlike group pair, incl. k_kl and gamma_kl)
            let eps_k = p.epsilon_k_kl[(i, j)];
            let sig = p.sigma_kl[(i, j)];
            let la = p.la_kl[(i, j)];
            let lr = p.lr_kl[(i, j)];
            let c = p.c_kl[(i, j)];

            let dij = (di + properties.diameter[j]) * 0.5;
            let (d3, [a1s_b_la, a1s_b_lr, a1s_b_2la, a1s_b_lalr, a1s_b_2lr]) =
                pair(i, j, dij, sig, la, lr);
            let pref = properties.segment_density * d3 * eps_k * 2.0 * PI * c;
            let a1_ij = pref * (a1s_b_la - a1s_b_lr);
            let a2_ij = pref * eps_k * c * k_hs * 0.5 * (a1s_b_2la - a1s_b_lalr * 2.0 + a1s_b_2lr);

            // note indices of f(i, alpha) are shifted due to 0-indexing.
            let alpha = parameters.alpha_kl[(i, j)];
            let a3_ij = -zeta_x_bar
                * f(3, alpha)
                * (zeta_x_bar * (zeta_x_bar * f(5, alpha) + f(4, alpha))).exp()
                * eps_k.powi(3);
            let xij = zeta_x_bar * f(0, alpha) + zx5 * f(1, alpha) + zx8 * f(2, alpha);

            let xs_ij = xs[i] * xs[j];
            // accumulate contributions
            a1 += a1_ij * xs_ij * 2.0;
            a2 += a2_ij * xs_ij * (xij + 1.0) * 2.0;
            a3 += a3_ij * xs_ij * 2.0;
        }
    }
    let rho = state.partial_density.sum();
    let a_disp = rho
        * properties.mean_segment_number
        * (a1 * t_inv + a2 * t_inv.powi(2) + a3 * t_inv.powi(3));

    // chain: molecules with Σ_k m_k,i = 1 contribute zero
    if p.m_i.iter().all(|&m| m == 1.0) {
        return (a_disp, D::zero());
    }

    // wrap rho_s in Dual to calculate da1/drho_s and da2/drho_s
    // for chain contribution on the fly.
    let rho_s_dual = Dual::from_re(properties.segment_density).derivative();
    let zeta_x_dual = if properties.segment_density.is_zero() {
        rho_s_dual * 0.0
    } else {
        Dual::from_re(zeta_x / properties.segment_density) * rho_s_dual
    };
    let k_hs_dual = (zeta_x_dual - 1.0).powi(4)
        / ((zeta_x_dual + zeta_x_dual.powi(2) - zeta_x_dual.powi(3)) * 4.0
            + zeta_x_dual.powi(4)
            + 1.0);

    let mut a_chain = D::zero();
    for (i, &m) in p.m_i.iter().enumerate() {
        if m == 1.0 {
            continue;
        }
        // molecular averages (Eq. 14-18); d̄^3 from the group diameters of molecule i
        let mut d3_bar = D::zero();
        for &k in &p.component_groups[i] {
            for &l in &p.component_groups[i] {
                let d_kl = (properties.diameter[k] + properties.diameter[l]) * 0.5;
                d3_bar += d_kl.powi(3) * (p.m[k] * p.m[l] / (m * m));
            }
        }
        let eps_k = p.epsilon_k_bar[i];
        let la = p.la_bar[i];
        let lr = p.lr_bar[i];
        let c = p.c_bar[i];
        let alpha = p.alpha_bar[i];

        // calculate a1 and a2 of the averaged pair using Dual(D)
        // (copied from saftvrmie::eos::dispersion, with x0 = σ̄/d̄)
        let d3 = Dual::from_re(d3_bar);
        let x0 = Dual::from_re(d3_bar.recip() * p.sigma3_bar[i]).cbrt();
        let pref = rho_s_dual * d3 * eps_k * 2.0 * PI * c;
        let [a1s_b_la, a1s_b_lr, a1s_b_2la, a1s_b_lalr, a1s_b_2lr] =
            a1s_b_terms(zeta_x_dual, x0, la, lr);
        let a1_ii = pref * (a1s_b_la - a1s_b_lr);
        let a2_ii = pref * eps_k * c * k_hs_dual * 0.5 * (a1s_b_2la - a1s_b_lalr * 2.0 + a1s_b_2lr);

        // use dual-parts of a1_ii and a2_ii for derivatives
        // and real-parts of a1s_b-terms.
        let x0 = x0.re;
        let pref = d3.re * eps_k * 2.0 * PI;
        let g_hs = (k[0] + k[1] * x0 + k[2] * x0.powi(2) + k[3] * x0.powi(3)).exp();
        let g1 = a1_ii.eps * 3.0 / pref - (a1s_b_la.re * la - a1s_b_lr.re * lr) * c;
        let g2_mca = a2_ii.eps * 3.0 / pref / eps_k
            - (a1s_b_2lr.re * lr - a1s_b_lalr.re * (la + lr) + a1s_b_2la.re * la)
                * k_hs_dual.re
                * c.powi(2);
        let beta_eps = t_inv * eps_k;
        let gamma = zeta_x_bar
            * beta_eps.exp_m1()
            * 10.0
            * (-(10.0 * (0.57 - alpha)).tanh() + 1.0)
            * (-zeta_x_bar * 6.7 - zeta_x_bar.powi(2) * 8.0).exp();
        let g2 = g2_mca * (gamma + 1.0);
        let ln_g_mie = g_hs.ln() + (beta_eps * g1 + beta_eps.powi(2) * g2) / g_hs;
        a_chain += -state.molefracs[i] * (m - 1.0) * ln_g_mie;
    }
    (a_disp, rho * a_chain)
}

#[inline]
pub(super) fn zeta_eff<D: DualNum<f64> + Copy>(zeta: D, lambda: f64) -> D {
    let li = 1. / lambda;
    let li2 = li * li;
    let li3 = li * li2;
    let c = [
        li * C[0][1] + li2 * C[0][2] + li3 * C[0][3] + C[0][0],
        li * C[1][1] + li2 * C[1][2] + li3 * C[1][3] + C[1][0],
        li * C[2][1] + li2 * C[2][2] + li3 * C[2][3] + C[2][0],
        li * C[3][1] + li2 * C[3][2] + li3 * C[3][3] + C[3][0],
    ];
    zeta * (zeta * (zeta * (zeta * c[3] + c[2]) + c[1]) + c[0])
}

/// Sutherland potential for mixtures (Eq. A 16) divided by 2 PI rho_s d_ij^3 epsilon_k_ij
#[inline]
fn a1s_ij<D: DualNum<f64> + Copy>(zeta_x: D, lambda: f64) -> D {
    let zeta_eff = zeta_eff(zeta_x, lambda);
    -(-zeta_eff * 0.5 + 1.0) / ((-zeta_eff + 1.0).powi(3) * (lambda - 3.0))
}

/// Eq. A 12 of Lafitte divided by 2 PI rho_s d_ij^3 epsilon_k_ij, with x0_l = x0^λ given
#[inline]
fn b_ij<D: DualNum<f64> + Copy>(zeta_x: D, x0: D, x0_l: D, lambda: f64) -> D {
    // x0^(3-λ) and x0^(4-λ) from x0^λ instead of two further powf calls
    let x0_3ml = x0.powi(3) / x0_l;
    let i = -(x0_3ml - 1.0) / (lambda - 3.0);
    let j = -(x0_3ml * x0 * (lambda - 3.0) - x0_3ml * (lambda - 4.0) - 1.0)
        / ((lambda - 3.0) * (lambda - 4.0));
    ((-zeta_x * 0.5 + 1.0) * i - zeta_x * (zeta_x + 1.0) * 4.5 * j) * (-zeta_x + 1.0).powi(-3)
}

/// Calculates x0^l (a1s_ij + b_ij) without prefactor C, with x0_l = x0^λ given
#[inline]
fn a1s_b_ij<D: DualNum<f64> + Copy>(zeta_x: D, x0: D, x0_l: D, lambda: f64) -> D {
    x0_l * (a1s_ij(zeta_x, lambda) + b_ij(zeta_x, x0, x0_l, lambda))
}

/// x0^λ (a1s + B) for the exponents λ^a, λ^r, 2λ^a, λ^a + λ^r and 2λ^r of a pair.
///
/// The only change to the kernels copied from `saftvrmie::eos::dispersion`: all powers of x0 are
/// obtained from x0^λ^a and x0^λ^r (two `powf` per pair instead of fifteen); equal up to round-off.
#[inline]
fn a1s_b_terms<D: DualNum<f64> + Copy>(zeta_x: D, x0: D, la: f64, lr: f64) -> [D; 5] {
    let (x0_la, x0_lr) = (x0.powf(la), x0.powf(lr));
    [
        a1s_b_ij(zeta_x, x0, x0_la, la),
        a1s_b_ij(zeta_x, x0, x0_lr, lr),
        a1s_b_ij(zeta_x, x0, x0_la * x0_la, 2.0 * la),
        a1s_b_ij(zeta_x, x0, x0_la * x0_lr, la + lr),
        a1s_b_ij(zeta_x, x0, x0_lr * x0_lr, 2.0 * lr),
    ]
}

fn f(k: usize, alpha: f64) -> f64 {
    let alpha2 = alpha * alpha;
    let alpha3 = alpha * alpha2;
    let phi = PHI[k];
    (alpha * phi[1] + alpha2 * phi[2] + alpha3 * phi[3] + phi[0])
        / (alpha * phi[4] + alpha2 * phi[5] + alpha3 * phi[6] + 1.0)
}

const C: [[f64; 4]; 4] = [
    [0.81096, 1.7888, -37.578, 92.284],
    [1.0205, -19.341, 151.26, -463.50],
    [-1.9057, 22.845, -228.14, 973.92],
    [1.0885, -6.1962, 106.98, -677.64],
];
