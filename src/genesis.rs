use std::{collections::BTreeMap, convert::TryInto};

use serde::{Deserialize, Serialize};
use tmelcrypt::{Ed25519PK, HashVal};

use crate::{
    melvm::Covenant, CoinData, CoinDataHeight, CoinID, Denom, NetID, SmtMapping, StakeDoc, State,
    TxHash, MICRO_CONVERTER,
};

/// Configuration of a genesis state. Serializable via serde.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenesisConfig {
    /// What kind of network?
    pub network: NetID,
    /// Initial supply of free money. This will be put at the zero-zero coin ID.
    pub init_coindata: CoinData,
    /// Mapping of initial stakeholders.
    pub stakes: BTreeMap<TxHash, StakeDoc>,
    /// Initial fee pool, in micromels. Half-life is approximately 15 days.
    pub init_fee_pool: u128,
}

impl GenesisConfig {
    /// The "standard" mainnet genesis.
    pub fn std_mainnet() -> Self {
        Self {
            network: NetID::Mainnet,
            init_coindata: CoinData {
                covhash: Covenant::std_ed25519_pk_legacy(Ed25519PK(
                    hex::decode("7323dcb65513b84470a76339cdf0062d47d82e205e834f2d7159684a0cb3b5ba")
                        .unwrap()
                        .try_into()
                        .unwrap(),
                ))
                .hash(),
                value: 1000000 * MICRO_CONVERTER, // 1 million SYM
                denom: Denom::Sym,
                additional_data: vec![],
            },
            stakes: ["7323dcb65513b84470a76339cdf0062d47d82e205e834f2d7159684a0cb3b5ba"]
                .iter()
                .map(|v| Ed25519PK(hex::decode(v).unwrap().try_into().unwrap()))
                .map(|pubkey| {
                    (
                        tmelcrypt::hash_single(&pubkey.0).into(), // A nonexistent hash
                        StakeDoc {
                            pubkey,
                            e_start: 0,
                            e_post_end: 3, // for the first two epochs (140 days)
                            syms_staked: 1,
                        },
                    )
                })
                .collect(),
            init_fee_pool: 6553600 * MICRO_CONVERTER, // 100 mel/day subsidy, decreasing rapidly
        }
    }

    /// The "standard" testnet genesis.
    pub fn std_testnet() -> Self {
        Self {
            network: NetID::Testnet,
            init_coindata: CoinData {
                covhash: Covenant::always_true().hash(),
                value: 1 << 32,
                denom: Denom::Mel,
                additional_data: vec![],
            },
            stakes: [
                "fae1ff56a62639c7959bf200465f4e06291e4e4dbd751cf4d2c13a8a6bea537c",
                "2ae54755b2e98a3059c68334af97b38603032be53bb2a1a3a183ae0f9d3bdaaf",
                "3aa3b5e2d64916a055da79635a4406999b66dfbe25afb10fa306aa01e42308a6",
                "85e374cc3e4dbf47b9a9697126e2e2ae90011b78a54b84adeb2ffe516b79769a",
            ]
            .iter()
            .map(|v| Ed25519PK(hex::decode(v).unwrap().try_into().unwrap()))
            .map(|pubkey| {
                (
                    tmelcrypt::hash_single(&pubkey.0).into(),
                    StakeDoc {
                        pubkey,
                        e_start: 0,
                        e_post_end: 1 << 32,
                        syms_staked: 1,
                    },
                )
            })
            .collect(),
            init_fee_pool: 1 << 64,
        }
    }

    /// Creates a [State] from this configuration.
    pub fn realize(self, db: &novasmt::Forest) -> State {
        let empty_tree = db.open_tree(HashVal::default().0).unwrap();
        let mut new_state = State {
            network: self.network,
            height: 0,
            history: SmtMapping::new(empty_tree.clone()),
            coins: SmtMapping::new(empty_tree.clone()),
            transactions: SmtMapping::new(empty_tree.clone()),
            fee_pool: self.init_fee_pool,
            fee_multiplier: MICRO_CONVERTER,
            tips: 0,

            dosc_speed: MICRO_CONVERTER,
            pools: SmtMapping::new(empty_tree.clone()),
            stakes: {
                let mut stakes = SmtMapping::new(empty_tree);
                for (k, v) in self.stakes.iter() {
                    stakes.insert(*k, *v);
                }
                stakes
            },
        };
        // init micromels etc
        new_state.coins.insert(
            CoinID::zero_zero(),
            CoinDataHeight {
                height: 0,
                coin_data: self.init_coindata,
            },
        );
        new_state
    }
}
