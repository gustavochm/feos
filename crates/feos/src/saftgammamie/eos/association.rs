//! Association strength of SAFT-γ Mie with the kernel of Dufal et al. (2015).
use crate::association::AssociationStrength;
use crate::saftgammamie::parameters::{SaftGammaMieAssociationRecord, SaftGammaMiePars};
use feos_core::StateHD;
use feos_core::parameter::{AssociationSite, BinaryParameters};
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;

/// Coefficients c_pq of the association kernel I (Haslam et al. (2020), Eq. 24), p + q <= 10.
///
/// From sgtpy (`gammamie_mixtures/association_aux.py`), Dufal et al. (2015) with the
/// corrigendum, Mol. Phys. 116 (2018) 283.
const C_PQ: [[f64; 11]; 11] = [
    [
        7.56425183020431e-02,
        -1.28667137050961e-01,
        1.28350632316055e-01,
        -7.25321780970292e-02,
        2.57782547511452e-02,
        -6.01170055221687e-03,
        9.33363147191978e-04,
        -9.55607377143667e-05,
        6.19576039900837e-06,
        -2.30466608213628e-07,
        3.74605718435540e-09,
    ],
    [
        1.34228218276565e-01,
        -1.82682168504886e-01,
        7.71662412959262e-02,
        -7.17458641164565e-04,
        -8.72427344283170e-03,
        2.97971836051287e-03,
        -4.84863997651451e-04,
        4.35262491516424e-05,
        -2.07789181640066e-06,
        4.13749349344802e-08,
        0.0,
    ],
    [
        -5.65116428942893e-01,
        1.00930692226792e+00,
        -6.60166945915607e-01,
        2.14492212294301e-01,
        -3.88462990166792e-02,
        4.06016982985030e-03,
        -2.39515566373142e-04,
        7.25488368831468e-06,
        -8.58904640281928e-08,
        0.0,
        0.0,
    ],
    [
        -3.87336382687019e-01,
        -2.11614570109503e-01,
        4.50442894490509e-01,
        -1.76931752538907e-01,
        3.17171522104923e-02,
        -2.91368915845693e-03,
        1.30193710011706e-04,
        -2.14505500786531e-06,
        0.0,
        0.0,
        0.0,
    ],
    [
        2.13713180911797e+00,
        -2.02798460133021e+00,
        3.36709255682693e-01,
        1.18106507393722e-03,
        -6.00058423301506e-03,
        6.26343952584415e-04,
        -2.03636395699819e-05,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        -3.00527494795524e-01,
        2.89920714512243e+00,
        -5.67134839686498e-01,
        5.18085125423494e-02,
        -2.39326776760414e-03,
        4.15107362643844e-05,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        -6.21028065719194e+00,
        -1.92883360342573e+00,
        2.84109761066570e-01,
        -1.57606767372364e-02,
        3.68599073256615e-04,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        1.16083532818029e+01,
        7.42215544511197e-01,
        -8.23976531246117e-02,
        1.86167650098254e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        -1.02632535542427e+01,
        -1.25035689035085e-01,
        1.14299144831867e-02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        4.65297446837297e+00,
        -1.92518067137033e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        -8.67296219639940e-01,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
];

/// Association kernel I(ρ_s σ_x³, kT/ε̄_ij) (Haslam et al. (2020), Eq. 24), Horner in both arguments.
pub(crate) fn association_kernel<D: DualNum<f64> + Copy>(rho_sigma3: D, t_red: D) -> D {
    C_PQ.iter().rev().fold(D::zero(), |acc, c_p| {
        acc * rho_sigma3 + c_p.iter().rev().fold(D::zero(), |a, &c| a * t_red + c)
    })
}

impl SaftGammaMiePars {
    /// Reduced segment density ρ_s σ_x³ of the association kernel (Haslam et al. (2020), Eq. 25),
    /// with σ_x³ = Σ_k Σ_l x_s,k x_s,l σ_kl³ (as ζ̄_x without π/6).
    pub(crate) fn reduced_density_sigma_x3<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D {
        let n = self.m.len();
        let x = self.component_index.map(|i| state.molefracs[i]);
        let m = self.m.map(D::from);
        let xs = x.component_mul(&m) / x.component_mul(&m).sum();
        let segment_density = self
            .component_index
            .map(|i| state.partial_density[i])
            .component_mul(&m)
            .sum();
        let mut sigma_x3 = D::zero();
        for k in 0..n {
            sigma_x3 += xs[k].powi(2) * self.sigma_kl[(k, k)].powi(3);
            for l in k + 1..n {
                sigma_x3 += xs[k] * xs[l] * self.sigma_kl[(k, l)].powi(3) * 2.0;
            }
        }
        segment_density * sigma_x3
    }
}

impl AssociationStrength for SaftGammaMiePars {
    type Record = SaftGammaMieAssociationRecord;

    /// F_kl,ab K_kl,ab (Haslam et al. (2020), Eq. 22-23) in units of Angstrom^3; the kernel I is
    /// applied in [SaftGammaMiePars::association_strength].
    fn association_strength_ij<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        _: usize,
        _: usize,
        assoc_ij: &Self::Record,
    ) -> D {
        (temperature.recip() * assoc_ij.epsilon_k_ab).exp_m1() * assoc_ij.kappa_ab
    }

    /// Δ = F K I with the kernel of the two molecules owning the sites; no contact value of the
    /// hard-sphere pair correlation function (Haslam et al. (2020), Eq. 22).
    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
        _: &DVector<D>,
        (sites1, sites2): (&[AssociationSite], &[AssociationSite]),
        association_parameters: &[BinaryParameters<Self::Record, ()>],
    ) -> DMatrix<D> {
        let mut delta = DMatrix::zeros(sites1.len(), sites2.len());
        if sites1.len() * sites2.len() == 0 {
            return delta;
        }

        let rho_sigma3 = self.reduced_density_sigma_x3(state);
        for bp in association_parameters {
            let [a, b] = [bp.id1, bp.id2];
            let [k, l] = [sites1[a].assoc_comp, sites2[b].assoc_comp];
            let [i, j] = [self.component_index[k], self.component_index[l]];
            let t_red = state.temperature / self.epsilon_k_bar_ij[(i, j)];
            delta[(a, b)] = association_kernel(rho_sigma3, t_red)
                * self.association_strength_ij(state.temperature, k, l, &bp.model_record);
        }
        delta
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::association::Association;
    use crate::hard_sphere::HardSphereProperties;
    use crate::saftgammamie::parameters::association_parameters;
    use crate::saftgammamie::test_utils::parameters;
    use approx::assert_relative_eq;

    /// The kernel as the plain double sum of Haslam et al. (2020), Eq. 24.
    fn kernel_sum(rho: f64, t: f64) -> f64 {
        (0..11)
            .flat_map(|p| (0..11 - p).map(move |q| (p, q)))
            .map(|(p, q)| C_PQ[p][q] * rho.powi(p as i32) * t.powi(q as i32))
            .sum()
    }

    #[test]
    fn kernel_horner_equals_sum() {
        for (rho, t) in [
            (0.0, 1.1),
            (0.86, 1.12),
            (0.3, 0.5),
            (1.2, 6.0),
            (0.6, 10.0),
        ] {
            // high powers of T* cancel (condition number ≈ 2e4 at T* = 10): both forms are accurate
            // to ≈ 1e-12 there, far better at physical T* ≲ 4
            assert_relative_eq!(
                association_kernel(rho, t),
                kernel_sum(rho, t),
                max_relative = 1e-11
            );
        }
        assert!(
            C_PQ.iter()
                .enumerate()
                .all(|(p, c)| c[11 - p..].iter().all(|&c| c == 0.0))
        );
    }

    fn ethanol_water(x_water: f64, rho: f64) -> (SaftGammaMiePars, StateHD<f64>) {
        let p = SaftGammaMiePars::new(&parameters(&["ethanol", "water"])).unwrap();
        let state = StateHD::new_density(
            320.0,
            &DVector::from_vec(vec![rho * (1.0 - x_water), rho * x_water]),
        );
        (p, state)
    }

    #[test]
    fn delta_is_f_k_i_without_contact_value() {
        let parameters = parameters(&["ethanol", "water"]);
        let a = association_parameters(&parameters).unwrap();
        let (p, state) = ethanol_water(0.4, 0.02);
        let d = p.hs_diameter(state.temperature);
        let delta = p.association_strength(&state, &d, (&a.sites_a, &a.sites_b), &a.binary_ab);
        // the diameters do not enter (no hard-sphere contact value)
        let delta_d =
            p.association_strength(&state, &(&d * 1.3), (&a.sites_a, &a.sites_b), &a.binary_ab);
        assert_eq!(delta, delta_d);

        let rho_sigma3 = p.reduced_density_sigma_x3(&state);
        for bp in &a.binary_ab {
            let [k, l] = [a.sites_a[bp.id1].assoc_comp, a.sites_b[bp.id2].assoc_comp];
            let [i, j] = [p.component_index[k], p.component_index[l]];
            let r = bp.model_record;
            let f = (r.epsilon_k_ab / state.temperature).exp() - 1.0;
            let t_red = state.temperature / p.epsilon_k_bar_ij[(i, j)];
            let expected = f * r.kappa_ab * kernel_sum(rho_sigma3, t_red);
            assert_relative_eq!(delta[(bp.id1, bp.id2)], expected, max_relative = 1e-12);
        }
        // the unlike molecular energy differs from both like ones (Eq. 26-27)
        let e = &p.epsilon_k_bar_ij;
        assert!(e[(0, 1)] != e[(0, 0)] && e[(0, 1)] != e[(1, 1)]);
    }

    #[test]
    fn zero_density_limit() {
        // Δ → F K I(0, T*): finite, no division by the density
        let parameters = parameters(&["ethanol", "water"]);
        let a = association_parameters(&parameters).unwrap();
        let (p, state) = ethanol_water(0.5, 1e-30);
        let d = p.hs_diameter(state.temperature);
        let delta = p.association_strength(&state, &d, (&a.sites_a, &a.sites_b), &a.binary_ab);
        for bp in &a.binary_ab {
            let [k, l] = [a.sites_a[bp.id1].assoc_comp, a.sites_b[bp.id2].assoc_comp];
            let t_red = state.temperature
                / p.epsilon_k_bar_ij[(p.component_index[k], p.component_index[l])];
            let r = bp.model_record;
            let expected =
                (r.epsilon_k_ab / state.temperature).exp_m1() * r.kappa_ab * kernel_sum(0.0, t_red);
            assert_relative_eq!(delta[(bp.id1, bp.id2)], expected, max_relative = 1e-12);
        }
    }

    #[test]
    fn analytic_equals_cross_association_solver() {
        // water: one A and one B site entry -> analytic solution in FeOs
        let parameters = parameters(&["water"]);
        let a = association_parameters(&parameters).unwrap();
        let p = SaftGammaMiePars::new(&parameters).unwrap();
        for rho in [1e-5, 0.01, 0.03] {
            let state = StateHD::new_density(350.0, &DVector::from_element(1, rho));
            let d = p.hs_diameter(state.temperature);
            let analytic = Association::new(50, 1e-12).helmholtz_energy_density(&p, &a, &state, &d);
            let general = Association::new_cross_association(50, 1e-12)
                .helmholtz_energy_density(&p, &a, &state, &d);
            assert_relative_eq!(analytic, general, max_relative = 1e-10);
        }
    }
}
