//! Properties with the groups of the sgtpy database (`saftgammamie::test_utils`).
use approx::assert_relative_eq;
use feos::saftgammamie::test_utils::{
    BinaryGroupRecord, binary_from_direct, groups, molecule, parameters,
};
use feos::saftgammamie::{SaftGammaMie, SaftGammaMieParameters};
use feos_core::parameter::{ChemicalRecord, Identifier};
use feos_core::{Contributions, PhaseEquilibrium, ResidualDyn, StateHD, Subset};
use nalgebra::{DVector, dvector};
use quantity::{KELVIN, PASCAL};

fn eos(names: &[&str]) -> SaftGammaMie {
    SaftGammaMie::new(parameters(names)).unwrap()
}

fn a_res(eos: &SaftGammaMie, state: &StateHD<f64>) -> f64 {
    eos.reduced_helmholtz_energy_density_contributions(state)
        .iter()
        .map(|(_, a)| a)
        .sum()
}

fn p_sat(eos: &SaftGammaMie, t: f64) -> f64 {
    let vle = PhaseEquilibrium::pure(&eos, t * KELVIN, None, Default::default()).unwrap();
    vle.vapor()
        .pressure(Contributions::Total)
        .convert_into(PASCAL)
}

#[test]
fn normal_boiling_points() {
    // published groups (Haslam et al. (2020)) at the experimental normal boiling points:
    // hexane +2.30 %, ethanol -1.99 %, water -1.78 % (typical deviations of the model are 1-3 %)
    for (name, t_b) in [("hexane", 341.88), ("ethanol", 351.44), ("water", 373.124)] {
        let p = p_sat(&eos(&[name]), t_b);
        assert_relative_eq!(p, 101325.0, max_relative = 0.03);
    }
}

#[test]
fn intramolecular_and_intermolecular_corrections() {
    // intramolecular CH3-CH2 record changes pure hexane
    let without =
        SaftGammaMieParameters::from_segments_hetero(vec![molecule("hexane")], &groups(), None);
    let without = SaftGammaMie::new(without.unwrap()).unwrap();
    let state = StateHD::new_density(350.0, &DVector::from_element(1, 0.004));
    assert!((a_res(&eos(&["hexane"]), &state) / a_res(&without, &state) - 1.0).abs() > 1e-3);

    // intermolecular records (CH3-H2O, CH2-H2O, CH2OH-H2O): pure components unchanged, mixture
    // changed; the first three records of `binary_groups` are the intramolecular ones
    let mixture = eos(&["ethanol", "water"]);
    let only_pure = SaftGammaMie::new(
        SaftGammaMieParameters::from_segments_hetero(
            vec![molecule("ethanol"), molecule("water")],
            &groups(),
            Some(&feos::saftgammamie::test_utils::binary_groups()[..3]),
        )
        .unwrap(),
    )
    .unwrap();
    for x in [dvector![0.01, 0.0], dvector![0.0, 0.03]] {
        let state = StateHD::new_density(320.0, &x);
        assert_relative_eq!(
            a_res(&mixture, &state),
            a_res(&only_pure, &state),
            max_relative = 1e-14
        );
    }
    let state = StateHD::new_density(320.0, &dvector![0.01, 0.02]);
    assert!((a_res(&mixture, &state) / a_res(&only_pure, &state) - 1.0).abs() > 1e-3);
}

#[test]
fn subset_keeps_intramolecular_pairs() {
    // FeOs uses subsets for pure-component references; they must keep the intramolecular records
    let mixture = eos(&["ethanol", "water"]);
    for (i, name) in [(0, "ethanol"), (1, "water")] {
        let pure = eos(&[name]);
        let subset = mixture.subset(&[i]);
        let state = StateHD::new_density(330.0, &DVector::from_element(1, 0.01));
        assert_eq!(a_res(&subset, &state), a_res(&pure, &state));
        assert_relative_eq!(
            p_sat(&subset, 330.0),
            p_sat(&pure, 330.0),
            max_relative = 1e-12
        );
    }
}

#[test]
fn merged_equals_unmerged() {
    // hexane as CH3 x2 + CH2 x4 and as six distinct groups with identical parameters
    let mut groups = groups();
    let ids = ["CH3", "CH2", "CH2_b", "CH2_c", "CH2_d", "CH3_b"];
    for id in &ids[2..] {
        let mut g = groups[if id.starts_with("CH3") { 0 } else { 1 }].clone();
        g.identifier = id.to_string();
        groups.push(g);
    }
    let ch3_ch2 = binary_from_direct(
        &groups[0].model_record,
        &groups[1].model_record,
        350.77,
        None,
    );
    let binary: Vec<_> = ids
        .iter()
        .flat_map(|a| ids.iter().map(move |b| (a, b)))
        .filter(|(a, b)| a < b && a.starts_with("CH3") != b.starts_with("CH3"))
        .map(|(a, b)| BinaryGroupRecord::new(a.to_string(), b.to_string(), Some(ch3_ch2)))
        .collect();
    let segments = ids.iter().map(|s| s.to_string()).collect();
    let unmerged = ChemicalRecord::new(Identifier::default(), segments, None);
    let unmerged =
        SaftGammaMieParameters::from_segments_hetero(vec![unmerged], &groups, Some(&binary));
    let unmerged = SaftGammaMie::new(unmerged.unwrap()).unwrap();
    let merged = eos(&["hexane"]);
    for (t, rho) in [(300.0, 1e-6), (350.0, 0.004), (450.0, 0.0055)] {
        let state = StateHD::new_density(t, &DVector::from_element(1, rho));
        let (cm, cu) = (
            merged.reduced_helmholtz_energy_density_contributions(&state),
            unmerged.reduced_helmholtz_energy_density_contributions(&state),
        );
        for ((nm, am), (nu, au)) in cm.iter().zip(&cu) {
            assert_eq!(nm, nu);
            assert_relative_eq!(am, au, max_relative = 1e-13);
        }
    }
    assert_relative_eq!(
        p_sat(&merged, 340.0),
        p_sat(&unmerged, 340.0),
        max_relative = 1e-10
    );
}

#[test]
fn contributions_and_chain_rule() {
    // water: one group with S = 1 -> m = 1, no chain; association present
    let state = StateHD::new_density(350.0, &DVector::from_element(1, 0.03));
    let names: Vec<_> = eos(&["water"])
        .reduced_helmholtz_energy_density_contributions(&state)
        .iter()
        .map(|(n, _)| *n)
        .collect();
    assert_eq!(names, ["Hard Sphere", "Dispersion", "Association"]);

    // a molecule of one CH3 group has m = 0.57 < 1: its (negative-prefactor) chain term is kept
    let methyl = ChemicalRecord::new(Identifier::default(), vec!["CH3".into()], None);
    let methyl =
        SaftGammaMieParameters::from_segments_hetero(vec![methyl], &groups(), None).unwrap();
    let methyl = SaftGammaMie::new(methyl).unwrap();
    let state = StateHD::new_density(300.0, &DVector::from_element(1, 0.01));
    let c = methyl.reduced_helmholtz_energy_density_contributions(&state);
    let chain = c.iter().find(|(n, _)| *n == "Chain").unwrap().1;
    assert!(chain.is_finite() && chain != 0.0);
}

#[test]
fn intramolecular_association_override() {
    // a diol HO-CH2-CH2-OH of two distinct CH2OH groups, with and without an override between them
    let mut groups = groups();
    let mut second = groups[2].clone();
    second.identifier = "CH2OH_b".into();
    groups.push(second);
    let diol = ChemicalRecord::new(
        Identifier::default(),
        vec!["CH2OH".into(), "CH2OH_b".into()],
        None,
    );
    let override_ = feos_core::parameter::BinaryAssociationRecord::with_id(
        "H".into(),
        "e1".into(),
        feos::saftgammamie::SaftGammaMieAssociationRecord::new(10.0, 3000.0),
    );
    let binary = [BinaryGroupRecord::with_association(
        "CH2OH".into(),
        "CH2OH_b".into(),
        None,
        vec![override_],
    )];
    let build = |binary: Option<&[BinaryGroupRecord]>| {
        let p = SaftGammaMieParameters::from_segments_hetero(vec![diol.clone()], &groups, binary);
        SaftGammaMie::new(p.unwrap()).unwrap()
    };
    let state = StateHD::new_density(400.0, &DVector::from_element(1, 0.008));
    let with = build(Some(&binary)).reduced_helmholtz_energy_density_contributions(&state);
    let without = build(None).reduced_helmholtz_energy_density_contributions(&state);
    assert_eq!(with[..3], without[..3]);
    assert!((with[3].1 / without[3].1 - 1.0).abs() > 1e-3);
}
