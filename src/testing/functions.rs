use crate::testing::utils::random_valid_txx;
use crate::{CoinData, CoinDataHeight, CoinID, CoinValue, Denom, GenesisConfig, MICRO_CONVERTER, StakeDoc, State, Transaction};
use crate::melvm::{Address, Covenant};

use std::collections::HashMap;

use novasmt::Forest;
use tmelcrypt::{Ed25519PK, Ed25519SK};

pub fn valid_txx(keypair: (Ed25519PK, Ed25519SK)) -> Vec<Transaction> {
    let (pk, sk) = keypair;
    let scr = Covenant::std_ed25519_pk_legacy(pk);
    let mut trng = rand::thread_rng();
    random_valid_txx(
        &mut trng,
        CoinID {
            txhash: tmelcrypt::HashVal([0; 32]).into(),
            index: 0,
        },
        CoinData {
            covhash: scr.hash(),
            value: (MICRO_CONVERTER * 1000).into(),
            denom: Denom::Mel,
            additional_data: vec![],
        },
        sk,
        &scr,
        1577000.into(),
    )
}

/// Create a state using a mapping from sk to syms staked for an epoch
pub fn create_state(stakers: &HashMap<Ed25519SK, CoinValue>, epoch_start: u64) -> State {
    // Create emtpy state
    let db: Forest = novasmt::Forest::new(novasmt::InMemoryBackend::default());
    let mut state: State = GenesisConfig::std_testnet().realize(&db);
    state.stakes.clear();

    // Insert a mel coin into state so we can transact
    let start_micromels: CoinValue = CoinValue(10000);
    let start_conshash: Address = Covenant::always_true().hash();
    state.coins.insert(
        CoinID {
            txhash: tmelcrypt::HashVal([0; 32]).into(),
            index: 0,
        },
        CoinDataHeight {
            coin_data: CoinData {
                covhash: start_conshash,
                value: start_micromels,
                denom: Denom::Mel,
                additional_data: Vec::new(),
            },
            height: 0.into(),
        },
    );

    // Insert data need for staking proofs
    stakers.iter().enumerate().for_each(|(index, (sk, syms_staked))| {
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