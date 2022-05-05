use std::{collections::BTreeMap, convert::TryInto};

use novasmt::ContentAddrStore;
use serde::{Deserialize, Serialize};
use themelio_structs::{
    BlockHeight, CoinData, CoinDataHeight, CoinID, CoinValue, Denom, NetID, StakeDoc, TxHash,
    MICRO_CONVERTER,
};
use tmelcrypt::{Ed25519PK, HashVal};

use crate::{melvm::Covenant, CoinMapping, SmtMapping, State};

/// Configuration of a genesis state. Serializable via serde.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenesisConfig {
    /// What kind of network?
    pub network: NetID,
    /// Initial supply of free money. This will be put at the zero-zero coin ID.
    pub init_coindata: CoinData,
    /// Mapping of initial stakeholders.
    pub stakes: BTreeMap<TxHash, StakeDoc>,
    /// Initial fee pool. Half-life is approximately 15 days.
    pub init_fee_pool: CoinValue,
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
                value: (1000000 * MICRO_CONVERTER).into(), // 1 million SYM
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
                            syms_staked: 1.into(),
                        },
                    )
                })
                .collect(),
            init_fee_pool: CoinValue::from_millions(6553600u64), // subsidy, decreasing rapidly
        }
    }

    /// The "standard" testnet genesis.
    pub fn std_testnet() -> Self {
        Self {
            network: NetID::Testnet,
            init_coindata: CoinData {
                covhash: Covenant::always_true().hash(),
                value: (1 << 32).into(),
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
                        syms_staked: 1.into(),
                    },
                )
            })
            .collect(),
            init_fee_pool: (1 << 64).into(),
        }
    }

    /// Creates a [State] from this configuration.
    pub fn realize<C: ContentAddrStore>(self, db: &novasmt::Database<C>) -> State<C> {
        let empty_tree = db.get_tree(HashVal::default().0).unwrap();
        let mut new_state = State {
            network: self.network,
            height: 0.into(),
            history: SmtMapping::new(empty_tree.clone()),
            coins: CoinMapping::new(empty_tree.clone()),
            transactions: Default::default(),
            fee_pool: self.init_fee_pool,
            fee_multiplier: MICRO_CONVERTER,
            tips: 0.into(),

            dosc_speed: MICRO_CONVERTER,
            pools: SmtMapping::new(empty_tree.clone()),
            stakes: {
                let mut stakes = SmtMapping::new(empty_tree);

                self.stakes.iter().for_each(|(key, value)| {
                    stakes.insert(*key, *value);
                });

                stakes
            },
        };
        // init micromels etc
        new_state.coins.insert_coin(
            CoinID::zero_zero(),
            CoinDataHeight {
                height: BlockHeight(0),
                coin_data: self.init_coindata,
            },
            new_state.tip_906(),
        );
        new_state
    }
}
