use std::{collections::BTreeMap, convert::TryInto};

use melstructs::{
    BlockHeight, CoinData, CoinDataHeight, CoinID, CoinValue, Denom, NetID, StakeDoc, TxHash,
    MICRO_CONVERTER,
};
use melvm::Covenant;
use novasmt::ContentAddrStore;
use serde::{Deserialize, Serialize};
use tip911_stakeset::StakeSet;
use tmelcrypt::{Ed25519PK, HashVal};

use crate::{genesis_balances, CoinMapping, SmtMapping, UnsealedState};

/// Configuration of a genesis state. Serializable via serde.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenesisConfig {
    /// The network identifier for the associated `State`. Once this is set, it cannot be changed by subsequent transactions.
    #[serde(with = "stdcode::asstr")]
    pub network: NetID,
    /// Initial supply of free money. This will be set to the zero-zero coin ID.
    pub init_coindata: CoinData,
    /// Mapping of initial stakeholders.
    pub stakes: BTreeMap<TxHash, StakeDoc>,
    /// Initial fee pool. Half-life is approximately 15 days.
    pub init_fee_pool: CoinValue,
    /// Initial fee multipliier
    pub init_fee_multiplier: u128,
}

impl GenesisConfig {
    /// The "standard" mainnet genesis.
    pub fn std_mainnet() -> Self {
        Self {
            network: NetID::Mainnet,
            init_coindata: CoinData {
                covhash: Covenant::std_ed25519_pk_legacy(Ed25519PK(
                    hex::decode("4ce983d241f1d40b0e5b65e0bd1a6877a35acaec5182f110810f1276103c829e")
                        .unwrap()
                        .try_into()
                        .unwrap(),
                ))
                .hash(),
                value: (1000000 * MICRO_CONVERTER).into(), // 1 million SYM
                denom: Denom::Sym,
                additional_data: Default::default(),
            },
            stakes: ["4ce983d241f1d40b0e5b65e0bd1a6877a35acaec5182f110810f1276103c829e"]
                .iter()
                .map(|v| Ed25519PK(hex::decode(v).unwrap().try_into().unwrap()))
                .map(|pubkey| {
                    (
                        tmelcrypt::hash_single(pubkey.0).into(), // A nonexistent hash
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
            init_fee_multiplier: MICRO_CONVERTER,
        }
    }

    /// The "standard" testnet genesis.
    pub fn std_testnet() -> Self {
        serde_yaml::from_slice(include_bytes!("genesis-testnet.yaml")).unwrap()
    }

    /// Creates an [UnsealedState] from this configuration.
    pub fn realize<C: ContentAddrStore>(self, db: &novasmt::Database<C>) -> UnsealedState<C> {
        let empty_tree = db.get_tree(HashVal::default().0).unwrap();
        let mut new_state = UnsealedState {
            network: self.network,
            height: 0.into(),
            history: SmtMapping::new(empty_tree.clone()),
            coins: CoinMapping::new(empty_tree.clone()),
            transactions: Default::default(),
            fee_pool: self.init_fee_pool,
            fee_multiplier: self.init_fee_multiplier,
            tips: 0.into(),

            dosc_speed: MICRO_CONVERTER,
            pools: SmtMapping::new(empty_tree),
            stakes: StakeSet::new(self.stakes.into_iter()),
        };
        // init micromels etc
        new_state.coins.insert_coin(
            CoinID::zero_zero(),
            CoinDataHeight {
                height: BlockHeight(0),
                coin_data: self.init_coindata,
            },
        );
        // add balances using faucet transactions in block 0
        new_state
            .apply_tx_batch(&genesis_balances::faucet_txs())
            .expect("error applying genesis balances");

        new_state
    }
}
