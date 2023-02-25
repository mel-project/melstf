use melstructs::BlockHeight;

/// TIP 901: change fee multiplier calculation. More details [here](https://github.com/themeliolabs/themelio-node/issues/26).
pub const TIP_901_HEIGHT: BlockHeight = BlockHeight(42700);

/// TIP 902: introduce non-MEL/non-MEL pools. More details [here](https://github.com/themeliolabs/themelio-node/issues/29).
pub const TIP_902_HEIGHT: BlockHeight = BlockHeight(180000);

/// TIP 906: coin count commitments. More details [here](https://github.com/themeliolabs/themelio-node/issues/61).
pub const TIP_906_HEIGHT: BlockHeight = BlockHeight(830000);

/// TIP 908: dense merkle trees for transactions. More details [here](https://github.com/themeliolabs/themelio-node/issues/69).
pub const TIP_908_HEIGHT: BlockHeight = BlockHeight(u64::MAX);

/// TIP 909: tokenomics. More details [here](https://github.com/themeliolabs/themelio-node/issues/86).
pub const TIP_909_HEIGHT: BlockHeight = BlockHeight(950000);

/// TIP 909a: tokenomics bugfix. More details [here](https://github.com/themeliolabs/themelio-node/issues/86#issuecomment-1097227054).
pub const TIP_909A_HEIGHT: BlockHeight = BlockHeight(1048000);
