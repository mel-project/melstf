use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Denom;

/// A key identifying a pool.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PoolKey {
    pub left: Denom,
    pub right: Denom,
}

impl PoolKey {
    /// Pool key with something and mel
    pub fn mel_and(other: Denom) -> Self {
        assert!(other != Denom::Mel);
        Self {
            left: other,
            right: Denom::Mel,
        }
        .to_canonical()
        .unwrap()
    }

    /// Ensures that this pool key is canonical. If the two denoms are the same, returns None.
    #[allow(clippy::comparison_chain)]
    pub fn to_canonical(&self) -> Option<Self> {
        if self.left.to_bytes() < self.right.to_bytes() {
            Some(Self {
                left: self.left,
                right: self.right,
            })
        } else if self.left.to_bytes() > self.right.to_bytes() {
            Some(Self {
                left: self.right,
                right: self.left,
            })
        } else {
            None
        }
    }

    /// Denomination of the pool liquidity token
    pub fn liq_token_denom(&self) -> Denom {
        Denom::Custom(tmelcrypt::hash_keyed(b"liq", self.to_bytes()).into())
    }

    fn to_bytes(&self) -> Vec<u8> {
        if self.left == Denom::Mel {
            self.right.to_bytes()
        } else if self.right == Denom::Mel {
            self.left.to_bytes()
        } else {
            let mut v = vec![0u8; 32];
            v.extend_from_slice(&stdcode::serialize(&(self.left, self.right)).unwrap());
            v
        }
    }

    pub fn from_bytes(vec: &[u8]) -> Option<Self> {
        if vec.len() > 32 {
            // first 32 bytes must all be zero
            if vec[..32] != [0u8; 32] {
                return None;
            }
            let lr: (Denom, Denom) = stdcode::deserialize(&vec).ok()?;
            Some(Self {
                left: lr.0,
                right: lr.1,
            })
        } else {
            Some(
                Self {
                    left: Denom::Mel,
                    right: Denom::from_bytes(vec)?,
                }
                .to_canonical()?,
            )
        }
    }
}

impl Serialize for PoolKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_bytes().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PoolKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let inner = <Vec<u8>>::deserialize(deserializer)?;
        PoolKey::from_bytes(&inner)
            .ok_or_else(|| serde::de::Error::custom("not the right format for a PoolKey"))
    }
}
