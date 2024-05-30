#![allow(clippy::upper_case_acronyms)]
#![doc = include_str!("../README.md")]

mod genesis;

mod smtmapping;
mod state;
pub mod stats;
mod testing;
mod init_balances;

pub use crate::genesis::*;
pub use crate::smtmapping::*;
pub use crate::state::melmint::*;
pub use crate::state::*;

#[cfg(test)]
#[macro_use]
extern crate lazy_static;
