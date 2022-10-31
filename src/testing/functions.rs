use crate::GenesisConfig;
use crate::{
    testing::constants::{DB, GENESIS_EPOCH_POST_END, GENESIS_EPOCH_START},
    State,
};

// const GENESIS_MEL_SUPPLY: u128 = 21_000_000;
// const GENESIS_NUM_STAKERS: u64 = 10;
// const GENESIS_EPOCH_START: u64 = 0;
// const GENESIS_EPOCH_POST_END: u64 = 1000;
// const GENESIS_STAKER_WEIGHT: u128 = 100;
// pub const SEND_MEL_AMOUNT: u128 = 30_000_000_000;

use crate::melvm::Covenant;

use std::collections::HashMap;

use novasmt::{Database, InMemoryCas};
use tap::Tap;
use themelio_structs::{
    Address, CoinData, CoinDataHeight, CoinID, CoinValue, Denom, NetID, StakeDoc, Transaction,
    MICRO_CONVERTER,
};
use tmelcrypt::{Ed25519PK, Ed25519SK};

use super::utils::random_valid_txx_count;

pub fn valid_txx(keypair: (Ed25519PK, Ed25519SK)) -> Vec<Transaction> {
    let (pk, sk) = keypair;
    let scr = Covenant::std_ed25519_pk_legacy(pk);
    let mut trng = rand::thread_rng();
    random_valid_txx_count(
        &mut trng,
        CoinID {
            txhash: tmelcrypt::HashVal([0; 32]).into(),
            index: 0,
        },
        CoinData {
            covhash: scr.hash(),
            value: (MICRO_CONVERTER * 10000).into(),
            denom: Denom::Mel,
            additional_data: vec![].into(),
        },
        sk,
        &scr,
        1577000.into(),
        100,
    )
}

/// Create a state using a mapping from sk to syms staked for an epoch
pub fn create_state(
    stakers: &HashMap<Ed25519SK, CoinValue>,
    epoch_start: u64,
) -> State<InMemoryCas> {
    // Create emtpy state
    let db = Database::new(InMemoryCas::default());
    let mut state = GenesisConfig::std_testnet()
        .tap_mut(|g| g.network = NetID::Custom02)
        .realize(&db);
    state.stakes.clear();

    // Insert a mel coin into state so we can transact
    let start_micromels: CoinValue = CoinValue(MICRO_CONVERTER * 10000);
    let start_conshash: Address = Covenant::always_true().hash();
    state.coins.insert_coin(
        CoinID {
            txhash: tmelcrypt::HashVal([0; 32]).into(),
            index: 0,
        },
        CoinDataHeight {
            coin_data: CoinData {
                covhash: start_conshash,
                value: start_micromels,
                denom: Denom::Mel,
                additional_data: Vec::new().into(),
            },
            height: 0.into(),
        },
        state.tip_906(),
    );

    // Insert data need for staking proofs
    stakers
        .iter()
        .enumerate()
        .for_each(|(index, (sk, syms_staked))| {
            state.stakes.insert(
                tmelcrypt::hash_single(&(index as u128).to_be_bytes()).into(),
                StakeDoc {
                    pubkey: sk.to_public(),
                    e_start: epoch_start,
                    e_post_end: 1000000000,
                    syms_staked: *syms_staked,
                },
            );
        });

    state
}

pub fn genesis_mel_coin_id() -> CoinID {
    CoinID::zero_zero()
}

/// Create a genesis state from mel coin and stakeholders
pub fn genesis_state(
    genesis_mel_coin_id: CoinID,
    genesis_mel_coin_data_height: CoinDataHeight,
    genesis_stakeholders: HashMap<(Ed25519PK, Ed25519SK), u128>,
) -> State<InMemoryCas> {
    // Init empty state with db reference
    let mut state = GenesisConfig::std_testnet().realize(&DB);

    // insert initial mel coin supply
    state.coins.insert_coin(
        genesis_mel_coin_id,
        genesis_mel_coin_data_height,
        state.tip_906(),
    );

    // Insert stake holders
    for (i, (&keypair, &syms_staked)) in genesis_stakeholders.iter().enumerate() {
        state.stakes.insert(
            tmelcrypt::hash_single(&(i as u64).to_be_bytes()).into(),
            StakeDoc {
                pubkey: keypair.0,
                e_start: GENESIS_EPOCH_START,
                e_post_end: GENESIS_EPOCH_POST_END,
                syms_staked: syms_staked.into(),
            },
        );
    }

    state
}
