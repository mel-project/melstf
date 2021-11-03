use crate::testing::utils::*;

use crate::MICRO_CONVERTER;
use crate::CoinData;
use crate::CoinID;
use crate::Transaction;
use crate::melvm::Covenant;
use crate::Denom;

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