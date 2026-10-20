//! Cost of one evaluation of the residual Helmholtz energy of SAFT-γ Mie for different dual
//! number types, compared with SAFT-VR Mie (single-group molecules) and gc-PC-SAFT (hexane),
//! and for merged, unmerged and atom-level-like (many distinct groups) systems.
use criterion::{Criterion, criterion_group, criterion_main};
use feos::gc_pcsaft::{GcPcSaft, GcPcSaftParameters};
use feos::saftgammamie::test_utils::{GroupRecord, groups, parameters};
use feos::saftgammamie::{SaftGammaMie, SaftGammaMieParameters, SaftGammaMieRecord};
use feos::saftvrmie::{SaftVRMie, test_utils::test_parameters};
use feos_core::parameter::{ChemicalRecord, Identifier, IdentifierOption};
use feos_core::{ResidualDyn, StateHD};
use nalgebra::DVector;
use num_dual::{Dual64, DualNum, HyperDual64};
use std::time::Duration;

fn a_res<D: DualNum<f64> + Copy, E: ResidualDyn>(eos: &E, state: &StateHD<D>) -> D {
    eos.reduced_helmholtz_energy_density_contributions(state)
        .iter()
        .fold(D::zero(), |acc, (_, a)| acc + *a)
}

/// f64, first (Dual64) and second (HyperDual64) density derivatives of the first component.
fn bench_eos<E: ResidualDyn>(c: &mut Criterion, name: &str, eos: &E, t: f64, rho: &[f64]) {
    let rho = DVector::from_column_slice(rho);
    let s0 = StateHD::new_density(t, &rho);
    let mut r1 = rho.map(Dual64::from);
    r1[0] = r1[0].derivative();
    let s1 = StateHD::new_density(Dual64::from(t), &r1);
    let mut r2 = rho.map(HyperDual64::from);
    r2[0] = r2[0].derivative1().derivative2();
    let s2 = StateHD::new_density(HyperDual64::from(t), &r2);
    let mut group = c.benchmark_group(name);
    group
        .measurement_time(Duration::from_secs(2))
        .sample_size(30);
    group.bench_function("f64", |b| b.iter(|| a_res(eos, &s0)));
    group.bench_function("dual", |b| b.iter(|| a_res(eos, &s1)));
    group.bench_function("hyperdual", |b| b.iter(|| a_res(eos, &s2)));
    group.finish();
}

/// A molecule of `n` distinct groups with the parameters of `template` (atom-level-like).
fn distinct_groups(prefix: &str, template: &[(&str, usize)]) -> (ChemicalRecord, Vec<GroupRecord>) {
    let base: Vec<_> = groups();
    let mut records = Vec::new();
    let mut segments = Vec::new();
    for (name, count) in template {
        let g = base.iter().find(|g| g.identifier == *name).unwrap();
        for i in 0..*count {
            let mut r = g.clone();
            r.identifier = format!("{prefix}:{name}:{i}");
            segments.push(r.identifier.clone());
            records.push(r);
        }
    }
    (
        ChemicalRecord::new(Identifier::default(), segments, None),
        records,
    )
}

fn saftgammamie(c: &mut Criterion) {
    // single group (ν* = m, S = 1) against SAFT-VR Mie with the same parameters
    let hexane_vr = test_parameters().remove("hexane").unwrap();
    let r = hexane_vr.pure[0].model_record.clone();
    let single = SaftGammaMieParameters::from_segments_hetero(
        vec![ChemicalRecord::new(
            Identifier::default(),
            vec!["G".into()],
            None,
        )],
        &[GroupRecord::new(
            "G".into(),
            1.0,
            SaftGammaMieRecord::new(r.m, 1.0, r.sigma, r.epsilon_k, r.lr, r.la),
        )],
        None,
    )
    .unwrap();
    bench_eos(
        c,
        "single_group/saftvrmie",
        &SaftVRMie::new(hexane_vr),
        350.0,
        &[0.004],
    );
    bench_eos(
        c,
        "single_group/saftgammamie",
        &SaftGammaMie::new(single).unwrap(),
        350.0,
        &[0.004],
    );

    // hexane: merged (2 groups), unmerged (6 distinct groups), gc-PC-SAFT (Sauer et al.)
    let merged = SaftGammaMie::new(parameters(&["hexane"])).unwrap();
    let (cr, records) = distinct_groups("hexane", &[("CH3", 2), ("CH2", 4)]);
    let unmerged = SaftGammaMieParameters::from_segments_hetero(vec![cr], &records, None).unwrap();
    let gc = GcPcSaftParameters::from_json_segments_hetero(
        &["hexane"],
        "../../parameters/pcsaft/gc_substances.json",
        "../../parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    bench_eos(c, "hexane/saftgammamie_merged", &merged, 350.0, &[0.004]);
    bench_eos(
        c,
        "hexane/saftgammamie_unmerged",
        &SaftGammaMie::new(unmerged).unwrap(),
        350.0,
        &[0.004],
    );
    bench_eos(c, "hexane/gc_pcsaft", &GcPcSaft::new(gc), 350.0, &[0.004]);

    // associating mixtures: ethanol + water (merged), and two atom-level-like alcohols of 20
    // distinct groups each (40 groups, 2 associating molecules)
    let ethanol_water = SaftGammaMie::new(parameters(&["ethanol", "water"])).unwrap();
    bench_eos(
        c,
        "associating/ethanol_water",
        &ethanol_water,
        330.0,
        &[0.01, 0.02],
    );
    let (cr1, mut g1) = distinct_groups("a", &[("CH3", 1), ("CH2", 18), ("CH2OH", 1)]);
    let (cr2, g2) = distinct_groups("b", &[("CH3", 1), ("CH2", 18), ("CH2OH", 1)]);
    g1.extend(g2);
    let large = SaftGammaMieParameters::from_segments_hetero(vec![cr1, cr2], &g1, None).unwrap();
    bench_eos(
        c,
        "associating/2x20_groups",
        &SaftGammaMie::new(large).unwrap(),
        400.0,
        &[0.001, 0.001],
    );
}

criterion_group!(bench, saftgammamie);
criterion_main!(bench);
