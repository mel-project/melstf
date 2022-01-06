use std::collections::HashMap;

use tmelcrypt::{Ed25519PK, Ed25519SK};

pub(in crate::testing) const GENESIS_MEL_SUPPLY: u128 = 21_000_000;
pub(in crate::testing) const GENESIS_NUM_STAKERS: u64 = 10;
pub(in crate::testing) const GENESIS_EPOCH_START: u64 = 0;
pub(in crate::testing) const GENESIS_EPOCH_POST_END: u64 = 1000;
pub(in crate::testing) const GENESIS_STAKER_WEIGHT: u128 = 100;
pub(in crate::testing) const SEND_MEL_AMOUNT: u128 = 30_000_000_000;

lazy_static! {
    pub static ref DB: novasmt::Database<novasmt::InMemoryCas> =
        novasmt::Database::new(novasmt::InMemoryCas::default());
    pub static ref GENESIS_COVENANT_KEYPAIR: (Ed25519PK, Ed25519SK) = tmelcrypt::ed25519_keygen();
    pub static ref GENESIS_STAKEHOLDERS: HashMap<(Ed25519PK, Ed25519SK), u128> = {
        let mut stakeholders = HashMap::new();

        let range = 0..GENESIS_NUM_STAKERS;

        range.into_iter().for_each(|_index| {
            stakeholders.insert(tmelcrypt::ed25519_keygen(), GENESIS_STAKER_WEIGHT);
        });

        stakeholders
    };
}
