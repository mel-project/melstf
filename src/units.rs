use std::fmt::Display;

use arbitrary::Arbitrary;
use derive_more::{Mul, MulAssign, Add, AddAssign, Display, Div, DivAssign, From, FromStr, Into, Sub, SubAssign};
use serde::{Deserialize, Serialize};

use crate::{MICRO_CONVERTER, STAKE_EPOCH};

/// Newtype representing a monetary value in microunits. The Display and FromStr implementations divide by 1,000,000 automatically.
#[derive(
    Arbitrary,
    Clone,
    Copy,
    Default,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    From,
    Into,
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Div,
    DivAssign,
    Mul,
    MulAssign,
)]
#[serde(transparent)]
pub struct CoinValue(pub u128);

impl Display for CoinValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{:06}",
            self.0 / MICRO_CONVERTER,
            self.0 % MICRO_CONVERTER
        )
    }
}

impl CoinValue {
    /// Converts from an integer value of millions of microunits.
    pub fn from_millions(i: impl Into<u64>) -> Self {
        let i: u64 = i.into();
        Self(i as u128 * MICRO_CONVERTER)
    }
}

/// Newtype representing a block height.
#[derive(
    Arbitrary,
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    From,
    Into,
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Display,
    FromStr,
    Div,
    DivAssign,
    Mul,
    MulAssign,
)]
#[serde(transparent)]
pub struct BlockHeight(pub u64);

impl BlockHeight {
    /// Epoch of this height
    pub fn epoch(&self) -> u64 {
        self.0 / STAKE_EPOCH
    }
}
