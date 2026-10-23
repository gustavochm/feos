//! Temperature-dependent terms of SAFT-γ Mie, computed once per temperature (as `temperature_aux` in sgtpy).
//!
//! A bubble or dew point at given temperature evaluates the Helmholtz energy hundreds of times at the same T.
//! The hard-sphere diameters (a quadrature) and the temperature-only factors of the monomer pair terms are the
//! same in all of these evaluations, so they are computed once and reused.
//!
//! Each quantity f(T) is stored as Taylor coefficients c_n = f⁽ⁿ⁾(T₀)/n!, n = 0..3, at T₀ = Re(T). With
//! δ = T − T₀, f(T) = Σ c_n δⁿ is exact for dual numbers up to third order: δ has no real part, so δ⁴ = 0. T₀ is
//! always the real part of the evaluated temperature (no extrapolation), and another temperature recomputes the
//! terms. Dual numbers of higher order (`NDERIV > 3`) use the direct expressions.
use crate::saftgammamie::parameters::SaftGammaMiePars;
use num_dual::{Dual3_64, DualNum};
use std::cell::RefCell;
use std::rc::Rc;

/// Taylor coefficients [f, f', f''/2, f'''/6] of a temperature-dependent quantity at T₀.
pub(crate) type Taylor = [f64; 4];

/// Temperature-only factors of a group pair (Lafitte et al. (2013), Eq. A12-A16).
pub(crate) struct PairTerms {
    /// d_kl³ in Å³
    pub d3: Taylor,
    /// x0^λ, x0^λ I_λ and x0^λ J_λ (x0 = σ_kl/d_kl) for λ = λa, λr, 2λa, λa + λr, 2λr
    pub x0_lambda: [Taylor; 5],
    pub x0_lambda_i: [Taylor; 5],
    pub x0_lambda_j: [Taylor; 5],
}

/// The temperature-dependent terms of one parameter set at one temperature T₀.
pub(crate) struct TemperatureTerms {
    /// Hard-sphere diameter d_k of each group in Å
    pub diameter: Vec<Taylor>,
    /// Pair terms for l ≥ k, stored as `pairs[k][l - k]`
    pub pairs: Vec<Vec<PairTerms>>,
}

thread_local! {
    /// The terms at the last temperature evaluated on this thread, keyed by (parameter id, bits of T₀)
    static CACHE: RefCell<Option<(usize, u64, Rc<TemperatureTerms>)>> = const { RefCell::new(None) };
}

impl TemperatureTerms {
    /// The terms of `parameters` at the real temperature `t0` (in K), computed on the first call at `t0`.
    pub(crate) fn get(parameters: &SaftGammaMiePars, t0: f64) -> Rc<Self> {
        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            match cache.as_ref() {
                Some((id, bits, terms)) if *id == parameters.id && *bits == t0.to_bits() => {
                    terms.clone()
                }
                _ => {
                    let terms = Rc::new(Self::new(parameters, t0));
                    *cache = Some((parameters.id, t0.to_bits(), terms.clone()));
                    terms
                }
            }
        })
    }

    /// Terms from one evaluation with third-order dual numbers in T.
    fn new(p: &SaftGammaMiePars, t0: f64) -> Self {
        let n = p.m.len();
        let t_inv = Dual3_64::from(t0).derivative().recip();
        let d: Vec<_> = (0..n).map(|k| p.hs_diameter_kk(k, t_inv)).collect();
        let pairs = (0..n)
            .map(|k| (k..n).map(|l| pair_terms(p, &d, k, l)).collect())
            .collect();
        // the value itself from the f64 quadrature, so that d_k equals the direct (and SAFT-VR Mie)
        // diameter bit for bit; dual-number arithmetic can round the real part differently
        let diameter = d
            .into_iter()
            .enumerate()
            .map(|(k, d)| {
                let mut c = taylor(d);
                c[0] = p.hs_diameter_kk(k, t0.recip());
                c
            })
            .collect();
        Self { diameter, pairs }
    }

    /// Terms of the pair (k, l), in either order.
    pub(crate) fn pair(&self, k: usize, l: usize) -> &PairTerms {
        let (k, l) = (k.min(l), k.max(l));
        &self.pairs[k][l - k]
    }
}

/// Temperature-only factors of the pair (k, l), with the parameters the monomer term uses (like pairs: the
/// group records, as in SAFT-VR Mie).
fn pair_terms(p: &SaftGammaMiePars, d: &[Dual3_64], k: usize, l: usize) -> PairTerms {
    let (sigma, la, lr) = if k == l {
        (p.sigma[k], p.la[k], p.lr[k])
    } else {
        (p.sigma_kl[(k, l)], p.la_kl[(k, l)], p.lr_kl[(k, l)])
    };
    let d_kl = (d[k] + d[l]) * 0.5;
    let x0 = d_kl.recip() * sigma;
    let (x0_la, x0_lr) = (x0.powf(la), x0.powf(lr));
    let powers = [
        (x0_la, la),
        (x0_lr, lr),
        (x0_la * x0_la, 2.0 * la),
        (x0_la * x0_lr, la + lr),
        (x0_lr * x0_lr, 2.0 * lr),
    ];
    // I_λ and J_λ of the B term (Lafitte et al. (2013), Eq. A12), as in dispersion_chain::b_ij
    let i_j = powers.map(|(x0_l, lambda)| {
        let x0_3ml = x0.powi(3) / x0_l;
        let i = -(x0_3ml - 1.0) / (lambda - 3.0);
        let j = -(x0_3ml * x0 * (lambda - 3.0) - x0_3ml * (lambda - 4.0) - 1.0)
            / ((lambda - 3.0) * (lambda - 4.0));
        (x0_l * i, x0_l * j)
    });
    PairTerms {
        d3: taylor(d_kl.powi(3)),
        x0_lambda: powers.map(|(x0_l, _)| taylor(x0_l)),
        x0_lambda_i: i_j.map(|(i, _)| taylor(i)),
        x0_lambda_j: i_j.map(|(_, j)| taylor(j)),
    }
}

fn taylor(f: Dual3_64) -> Taylor {
    [f.re, f.v1, f.v2 / 2.0, f.v3 / 6.0]
}

/// Powers δ, δ², δ³ of δ = T − Re(T), shared by all cached quantities of one evaluation.
pub(crate) struct Delta<D>([D; 3]);

impl<D: DualNum<f64> + Copy> Delta<D> {
    pub(crate) fn new(temperature: D) -> Self {
        let delta = temperature - temperature.re();
        Self([delta, delta * delta, delta * delta * delta])
    }

    /// f(T) from its Taylor coefficients at Re(T)
    pub(crate) fn eval(&self, c: &Taylor) -> D {
        let [d1, d2, d3] = self.0;
        d3 * c[3] + d2 * c[2] + d1 * c[1] + c[0]
    }
}
