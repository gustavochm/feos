//! Reduction to SAFT-VR Mie: molecules of one group with ν* = m and S = 1 (MASTERPLAN_GAMMAMIE C.2).
use approx::assert_relative_eq;
use feos::saftgammamie::{SaftGammaMie, SaftGammaMieParameters, SaftGammaMieRecord};
use feos::saftvrmie::{SaftVRMie, SaftVRMieParameters, test_utils::test_parameters};
use feos_core::parameter::{
    ChemicalRecord, Identifier, IdentifierOption, PureRecord, SegmentRecord,
};
use feos_core::{Contributions, PhaseEquilibrium, ResidualDyn, State, StateHD};
use nalgebra::DVector;
use nalgebra::dvector;
use quantity::{KELVIN, METER, MOL};
use std::f64::consts::FRAC_PI_6;

/// Small deterministic random number generator (xorshift64*) for random states.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    /// Uniform number in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Log-uniform number in [lo, hi).
    pub fn log_range(&mut self, lo: f64, hi: f64) -> f64 {
        self.range(lo.ln(), hi.ln()).exp()
    }
}

const NON_ASSOCIATING: [&str; 6] = [
    "methane",
    "ethane",
    "hexane",
    "decane",
    "carbon dioxide",
    "perfluorohexane",
];

/// SAFT-VR Mie and SAFT-γ Mie (one group per molecule) for the given Lafitte molecules.
pub fn single_group_models(names: &[&str]) -> (SaftVRMie, SaftGammaMie) {
    let lafitte = test_parameters();
    let mut pure_records = Vec::new();
    let mut groups = Vec::new();
    let mut molecules = Vec::new();
    for &name in names {
        let p = &lafitte[name];
        let r = p.pure[0].model_record.clone();
        let mw = 1.0; // molar weight does not enter the residual Helmholtz energy
        let identifier = Identifier::new(None, Some(name), None, None, None, None);
        pure_records.push(PureRecord::new(identifier.clone(), mw, r.clone()));
        groups.push(SegmentRecord::new(
            name.to_string(),
            mw,
            SaftGammaMieRecord::new(r.m, 1.0, r.sigma, r.epsilon_k, r.lr, r.la),
        ));
        molecules.push(ChemicalRecord::new(
            identifier,
            vec![name.to_string()],
            None,
        ));
    }
    let vrmie =
        SaftVRMieParameters::from_records(pure_records, vec![], IdentifierOption::Name).unwrap();
    let gammamie = SaftGammaMieParameters::from_segments_hetero(molecules, &groups, None).unwrap();
    (SaftVRMie::new(vrmie), SaftGammaMie::new(gammamie).unwrap())
}

/// Random states: T in [200, 600] K, packing fraction (with σ) log-uniform in [1e-5, 0.45].
pub fn random_states(eos: &SaftVRMie, n: usize, seed: u64) -> Vec<StateHD<f64>> {
    let mut rng = Rng::new(seed);
    let p = &eos.params;
    let nc = p.m.len();
    (0..n)
        .map(|_| {
            let temperature = rng.range(200.0, 600.0);
            let x = DVector::from_fn(nc, |_, _| rng.range(0.05, 1.0));
            let x = &x / x.sum();
            let eta = rng.log_range(1e-5, 0.45);
            let msigma3: f64 = (0..nc).map(|i| x[i] * p.m[i] * p.sigma[i].powi(3)).sum();
            let rho = eta / (FRAC_PI_6 * msigma3);
            StateHD::new_density(temperature, &(x * rho))
        })
        .collect()
}

fn contributions(eos: &impl ResidualDyn, state: &StateHD<f64>) -> Vec<(&'static str, f64)> {
    eos.reduced_helmholtz_energy_density_contributions(state)
}

fn check_contributions(names: &[&str], seed: u64) {
    let (vrmie, gammamie) = single_group_models(names);
    for state in random_states(&vrmie, 50, seed) {
        let v = contributions(&vrmie, &state);
        let g = contributions(&gammamie, &state);
        let get = |c: &[(&str, f64)], name| c.iter().find(|(n, _)| *n == name).map(|(_, a)| *a);
        let scale = state.partial_density.sum();
        let hs_v = get(&v, "Hard Sphere").unwrap();
        assert_relative_eq!(
            get(&g, "Hard Sphere").unwrap(),
            hs_v,
            max_relative = 1e-12,
            epsilon = 1e-12 * scale
        );
        let disp_chain_v = get(&v, "Dispersion + Chain")
            .or(get(&v, "Dispersion"))
            .unwrap();
        let disp_chain_g = get(&g, "Dispersion").unwrap() + get(&g, "Chain").unwrap_or(0.0);
        assert_relative_eq!(
            disp_chain_g,
            disp_chain_v,
            max_relative = 1e-12,
            epsilon = 1e-12 * scale
        );
    }
}

#[test]
fn contributions_pure() {
    for (seed, name) in NON_ASSOCIATING.iter().enumerate() {
        check_contributions(&[name], seed as u64 + 1);
    }
}

#[test]
fn contributions_ternary() {
    check_contributions(&["hexane", "decane", "carbon dioxide"], 42);
}

#[test]
fn methane_has_no_chain() {
    let (_, gammamie) = single_group_models(&["methane"]);
    let state = StateHD::new_density(300.0, &DVector::from_element(1, 0.01));
    let names: Vec<_> = contributions(&gammamie, &state)
        .iter()
        .map(|(n, _)| *n)
        .collect();
    assert_eq!(names, ["Hard Sphere", "Dispersion"]);
}

#[test]
fn state_properties() {
    // pressure and residual chemical potentials of the ternary through FeOs' State
    let (vrmie, gammamie) = single_group_models(&["hexane", "decane", "carbon dioxide"]);
    let (vrmie, gammamie) = (&vrmie, &gammamie);
    let moles = dvector![0.3, 0.2, 0.5] * MOL;
    for (t, v) in [(300.0, 1e-3), (450.0, 1e-4), (350.0, 2e-4)] {
        let s1 = State::new_nvt(&vrmie, t * KELVIN, v * METER.powi::<3>(), &moles).unwrap();
        let s2 = State::new_nvt(&gammamie, t * KELVIN, v * METER.powi::<3>(), &moles).unwrap();
        assert_relative_eq!(
            s2.pressure(Contributions::Total),
            s1.pressure(Contributions::Total),
            max_relative = 1e-12
        );
        let (m1, m2) = (
            s1.residual_chemical_potential(),
            s2.residual_chemical_potential(),
        );
        for i in 0..3 {
            assert_relative_eq!(m2.get(i), m1.get(i), max_relative = 1e-11);
        }
    }
}

#[test]
fn phase_equilibria() {
    for name in ["ethane", "hexane", "carbon dioxide"] {
        let (vrmie, gammamie) = single_group_models(&[name]);
        let (vrmie, gammamie) = (&vrmie, &gammamie);
        let opt = Default::default();
        let c1: State<_> = State::critical_point(&vrmie, (), None, None, opt).unwrap();
        let c2: State<_> = State::critical_point(&gammamie, (), None, None, opt).unwrap();
        assert_relative_eq!(c2.temperature, c1.temperature, max_relative = 1e-10);
        assert_relative_eq!(
            c2.pressure(Contributions::Total),
            c1.pressure(Contributions::Total),
            max_relative = 1e-10
        );
        let t = 0.7 * c1.temperature;
        let p1 = PhaseEquilibrium::pure(&vrmie, t, None, opt)
            .unwrap()
            .vapor()
            .pressure(Contributions::Total);
        let p2 = PhaseEquilibrium::pure(&gammamie, t, None, opt)
            .unwrap()
            .vapor()
            .pressure(Contributions::Total);
        assert_relative_eq!(p2, p1, max_relative = 1e-10);
    }
}
