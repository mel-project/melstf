use std::path::Path;

use novasmt::Database;
use themelio_stf::{melvm::Covenant, GenesisConfig};
use themelio_structs::{CoinData, CoinValue, Denom, NetID, TransactionBuilder, TxKind};

fn main() {
    let test_state = GenesisConfig {
        network: NetID::Custom02,
        init_coindata: CoinData {
            value: CoinValue(10000),
            denom: Denom::Mel,
            additional_data: vec![],
            covhash: Covenant::always_true().hash(),
        },
        stakes: Default::default(),
        init_fee_pool: CoinValue(0),
    }
    .realize(&Database::new(MeshaCas::new(
        meshanina::Mapping::open(Path::new("/tmp/test.db")).unwrap(),
    )));
    // test basic transactions
    loop {
        let test_tx = TransactionBuilder::new()
            .kind(TxKind::Faucet)
            .fee(CoinValue::from_millions(1u64))
            .build()
            .unwrap();
    }
}

use ethnum::U256;
use novasmt::ContentAddrStore;

/// A meshanina-backed autosmt backend
pub struct MeshaCas {
    inner: meshanina::Mapping,
}

impl MeshaCas {
    /// Takes exclusively ownership of a Meshanina database and creates an autosmt backend.
    pub fn new(db: meshanina::Mapping) -> Self {
        Self { inner: db }
    }

    /// Syncs to disk.
    pub fn flush(&self) {
        self.inner.flush()
    }
}

impl ContentAddrStore for MeshaCas {
    fn get<'a>(&'a self, key: &[u8]) -> Option<std::borrow::Cow<'a, [u8]>> {
        self.inner
            .get(U256::from_le_bytes(tmelcrypt::hash_single(key).0))
    }

    fn insert(&self, key: &[u8], value: &[u8]) {
        self.inner
            .insert(U256::from_le_bytes(tmelcrypt::hash_single(key).0), value)
    }
}
