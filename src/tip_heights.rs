use themelio_structs::BlockHeight;

/// TIP 901: change fee multiplier calculation
pub const TIP_901_HEIGHT: BlockHeight = BlockHeight(42700);

/// TIP 902: introduce non-MEL/non-MEL pools
pub const TIP_902_HEIGHT: BlockHeight = BlockHeight(180000);

/// TIP 906: coin count commitments
pub const TIP_906_HEIGHT: BlockHeight = BlockHeight(830000);

/// TIP 908: dense merkle trees for transactions
pub const TIP_908_HEIGHT: BlockHeight = BlockHeight(u64::MAX);

/// TIP 909: tokenomics
pub const TIP_909_HEIGHT: BlockHeight = BlockHeight(950000);

/// TIP 909a: tokenomics bugfix
pub const TIP_909A_HEIGHT: BlockHeight = BlockHeight(1048000);
