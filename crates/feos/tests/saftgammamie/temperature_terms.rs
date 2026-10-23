//! Temperature terms computed once per temperature (`eos::temperature`) against the direct expressions,
//! which are used for dual numbers of higher order than three.
use approx::assert_relative_eq;
use feos::saftgammamie::SaftGammaMie;
use feos::saftgammamie::test_utils::parameters;
use feos_core::{ResidualDyn, StateHD, Subset};
use nalgebra::{DVector, dvector};
use num_dual::{Dual2_64, Dual3, Dual3_64, Dual64, DualNum, HyperDual64};

fn a_res<D: DualNum<f64> + Copy>(eos: &SaftGammaMie, t: D, rho: &DVector<D>) -> D {
    let state = StateHD::new_density(t, rho);
    eos.reduced_helmholtz_energy_density_contributions(&state)
        .iter()
        .fold(D::zero(), |acc, (_, a)| acc + *a)
}

/// Direct expressions (`NDERIV` = 4): T derivatives up to third order in the outer dual number, the first
/// derivative with respect to the density of component 0 in the inner one.
fn direct(eos: &SaftGammaMie, t: f64, rho: &DVector<f64>) -> Dual3<Dual64, f64> {
    let t = Dual3::from_re(Dual64::from_re(t)).derivative();
    let mut r = rho.map(|r| Dual3::from_re(Dual64::from_re(r)));
    r[0] = Dual3::from_re(Dual64::from_re(rho[0]).derivative());
    a_res(eos, t, &r)
}

#[test]
fn cached_terms_equal_direct_expressions() {
    let systems: [&[&str]; 3] = [
        &["hexane", "ethanol", "water"],
        &["hexane", "ethanol"],
        &["water"],
    ];
    for names in systems {
        let eos = SaftGammaMie::new(parameters(names)).unwrap();
        let rho = DVector::from_fn(names.len(), |i, _| 0.002 * (i + 1) as f64);
        for t in [250.0, 330.0, 600.0] {
            let r = direct(&eos, t, &rho);
            assert_relative_eq!(a_res(&eos, t, &rho), r.re.re, max_relative = 1e-13);

            // temperature derivatives
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
            let d3 = a_res(
                &eos,
                Dual3_64::from_re(t).derivative(),
                &rho.map(Dual3_64::from_re),
            );
            assert_relative_eq!(d1.eps, r.v1.re, max_relative = 1e-12);
            assert_relative_eq!(d2.v2, r.v2.re, max_relative = 1e-11);
            assert_relative_eq!(d3.v3, r.v3.re, max_relative = 1e-10);

            // density and mixed temperature-density derivatives
            let mut r1 = rho.map(Dual64::from_re);
            r1[0] = r1[0].derivative();
            let mut rh = rho.map(HyperDual64::from_re);
            rh[0] = rh[0].derivative2();
            let h = a_res(&eos, HyperDual64::from_re(t).derivative1(), &rh);
            assert_relative_eq!(
                a_res(&eos, Dual64::from_re(t), &r1).eps,
                r.re.eps,
                max_relative = 1e-12
            );
            assert_relative_eq!(h.eps1eps2, r.v1.eps, max_relative = 1e-12);
        }
    }
}

#[test]
fn cache_follows_parameters_and_temperature() {
    let a = SaftGammaMie::new(parameters(&["hexane", "ethanol"])).unwrap();
    let b = SaftGammaMie::new(parameters(&["ethanol", "water"])).unwrap();
    let rho = dvector![0.003, 0.006];
    let first = a_res(&a, 300.0, &rho);

    // alternating parameter sets and temperatures: every evaluation uses its own terms
    for (eos, t) in [
        (&b, 300.0),
        (&a, 350.0),
        (&b, 350.0),
        (&a, 300.0 + 1e-9),
        (&a, 300.0),
    ] {
        assert_relative_eq!(
            a_res(eos, t, &rho),
            direct(eos, t, &rho).re.re,
            max_relative = 1e-13
        );
    }
    assert_eq!(a_res(&a, 300.0, &rho), first);

    // a subset gets new terms
    let ethanol = SaftGammaMie::new(parameters(&["ethanol"])).unwrap();
    let rho1 = dvector![0.005];
    assert_relative_eq!(
        a_res(&a.subset(&[1]), 320.0, &rho1),
        a_res(&ethanol, 320.0, &rho1),
        max_relative = 1e-14
    );
}
