#![allow(clippy::upper_case_acronyms)]

//! This crate contains the data structures and core algorithms that comprise Themelio's core state machine.
//! Any piece of software needing to parse Themelio data, validate Themelio transactions, or answer questions like
//! "what happens to the Themelio state if transactions A, B, and C happen" can use this minimal-depedency crate.
//!
//! Roughly, the structs in this crate are organized as follows:
//! - `State` represents a full Themelio world-state and it's not directly serializable. It includes *all* the information needed to validate new transactions and blocks, such as a SMT of all outstanding coins, Melmint parameters, etc. It has methods taking `Transaction`s etc that advance the state, as well as others to produce serializable blocks, headers, etc.
//! - `Transaction` represents a serializable Themelio transaction. It has some helper methods to count coins, estimate fees, etc, largely to help build wallets.
//! - `StakeDoc`, which every `State` includes, encapsulates the Symphonia epoch-based stake information.
//! - `SmtMapping` represents a type-safe SMT-backed mapping that is extensively used within the crate.
mod genesis;
pub mod melpow;
pub mod melvm;
mod smtmapping;
mod stake;
mod state;
mod testing;

pub use crate::genesis::*;
pub use crate::smtmapping::*;
pub use crate::state::melmint::*;
pub use crate::state::*;

use std::ops::{Deref, DerefMut};

use arbitrary::Arbitrary;
use serde::{Deserialize, Serialize};

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

#[derive(
    Arbitrary, Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default,
)]
/// A type that wraps a bytevector, serializing as hexadecimal for JSON.
#[serde(transparent)]
pub struct HexBytes(#[serde(with = "stdcode::hex")] pub Vec<u8>);

impl Deref for HexBytes {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for HexBytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Vec<u8>> for HexBytes {
    fn from(val: Vec<u8>) -> Self {
        Self(val)
    }
}
