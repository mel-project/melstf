use tmelcrypt::HashVal;

use crate::melvm::CovHash;

/// Maximum coin value
pub const MAX_COINVAL: u128 = 1 << 120;

/// 1e6
pub const MICRO_CONVERTER: u128 = 1_000_000;

/// Coin destruction covhash
pub const COVHASH_DESTROY: CovHash = CovHash(HashVal([0; 32]));

/// A stake epoch is 200,000 blocks.
pub const STAKE_EPOCH: u64 = 200000;
