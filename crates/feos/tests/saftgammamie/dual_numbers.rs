//! Derivatives with dual numbers against central finite differences.
use approx::assert_relative_eq;
use feos::hard_sphere::HardSphereProperties;
use feos::saftgammamie::SaftGammaMie;
use feos::saftgammamie::test_utils::parameters;
use feos_core::{ResidualDyn, StateHD};
use nalgebra::{DVector, dvector};
use num_dual::{Dual2_64, Dual64, DualNum};

fn a_res<D: DualNum<f64> + Copy>(eos: &SaftGammaMie, t: D, rho: &DVector<D>) -> D {
    let state = StateHD::new_density(t, rho);
    eos.reduced_helmholtz_energy_density_contributions(&state)
        .iter()
        .fold(D::zero(), |acc, (_, a)| acc + *a)
}

#[test]
fn derivatives_against_finite_differences() {
    let eos = SaftGammaMie::new(parameters(&["hexane", "ethanol", "water"])).unwrap();
    let t = 330.0;
    let rho = dvector![0.002, 0.004, 0.01];
    let f = |t: f64, rho: &DVector<f64>| a_res(&eos, t, rho);

    // temperature: first derivative (Dual64) and second derivative (Dual2_64)
    let h = 1e-3;
    let fd1 = (f(t + h, &rho) - f(t - h, &rho)) / (2.0 * h);
    let fd2 = (f(t + h, &rho) - 2.0 * f(t, &rho) + f(t - h, &rho)) / (h * h);
    let d1 = a_res(
        &eos,
        Dual64::from_re(t).derivative(),
        &rho.map(Dual64::from_re),
    );
    let d2 = a_res(
        &eos,
        Dual2_64::from_re(t).derivative(),
        &rho.map(Dual2_64::from_re),
    );
    assert_relative_eq!(d1.eps, fd1, max_relative = 1e-6);
    assert_relative_eq!(d2.v1, d1.eps, max_relative = 1e-13);
    assert_relative_eq!(d2.v2, fd2, max_relative = 1e-4);

    // partial densities (association and chain depend on all of them)
    for i in 0..3 {
        let h = 1e-7;
        let mut rp = rho.clone();
        let mut rm = rho.clone();
        rp[i] += h;
        rm[i] -= h;
        let fd1 = (f(t, &rp) - f(t, &rm)) / (2.0 * h);
        let fd2 = (f(t, &rp) - 2.0 * f(t, &rho) + f(t, &rm)) / (h * h);
        let mut r1 = rho.map(Dual64::from_re);
        r1[i] = r1[i].derivative();
        let mut r2 = rho.map(Dual2_64::from_re);
        r2[i] = r2[i].derivative();
        let d1 = a_res(&eos, Dual64::from_re(t), &r1);
        let d2 = a_res(&eos, Dual2_64::from_re(t), &r2);
        assert_relative_eq!(d1.eps, fd1, max_relative = 1e-6);
        assert_relative_eq!(d2.v2, fd2, max_relative = 1e-3);
    }
}

#[test]
fn diameter_temperature_derivative() {
    let eos = SaftGammaMie::new(parameters(&["hexane", "ethanol", "water"])).unwrap();
    let t = 330.0;
    let h = 1e-3;
    let d = eos.params.hs_diameter(Dual64::from_re(t).derivative());
    let (dp, dm) = (eos.params.hs_diameter(t + h), eos.params.hs_diameter(t - h));
    for k in 0..d.len() {
        assert_relative_eq!(d[k].eps, (dp[k] - dm[k]) / (2.0 * h), max_relative = 1e-7);
    }
}
