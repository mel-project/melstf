use crate::{BlockHeight, CoinValue};

/// Maximum coin value
pub const MAX_COINVAL: CoinValue = CoinValue(1 << 120);

/// 1e6
pub const MICRO_CONVERTER: u128 = 1_000_000;

/// A stake epoch is 200,000 blocks.
pub const STAKE_EPOCH: u64 = 200000;

/// TIP 901: change fee multiplier calculation
pub const TIP_901_HEIGHT: BlockHeight = BlockHeight(42700);

/// TIP 902: introduce non-MEL/non-MEL pools
pub const TIP_902_HEIGHT: BlockHeight = BlockHeight(180000);
