//! Group-contribution Statistical Associating Fluid Theory for Mie potentials (SAFT-γ Mie)
//!
//! - [Papaioannou et al. (2014)](https://doi.org/10.1063/1.4851455)
//! - [Haslam et al. (2020)](https://doi.org/10.1021/acs.jced.0c00746)
//! - [Dufal et al. (2015)](https://doi.org/10.1080/00268976.2015.1029027) (association kernel)
pub(crate) mod eos;
pub(crate) mod parameters;

pub use eos::{SaftGammaMie, SaftGammaMieOptions};

pub use parameters::{
    SaftGammaMieAssociationRecord, SaftGammaMieBinaryRecord, SaftGammaMieParameters,
    SaftGammaMiePars, SaftGammaMieRecord, association_parameters, test_utils,
};
