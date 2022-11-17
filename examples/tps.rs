use std::{path::Path, time::Instant};

use ethnum::U256;
use novasmt::ContentAddrStore;
use once_cell::sync::Lazy;
use themelio_stf::{melvm::Covenant, GenesisConfig, State};
use themelio_structs::{Address, CoinData, Denom, NetID, Transaction, TxKind};

fn main() {
    let mut state = zerofee_state().seal(None).next_state();
    for (i, tx) in TEST_INPUT.iter().enumerate() {
        let tx_start = Instant::now();
        state.apply_tx(tx).unwrap();
        println!("apply tx {} took: {:?}", i, tx_start.elapsed());
        println!("state now {}", state.coins.root_hash());
    }
}

fn generate_txx(n: usize) -> Vec<Transaction> {
    let fixed_output = CoinData {
        covhash: Covenant::always_true().hash(),
        value: 100.into(),
        denom: Denom::Mel,
        additional_data: vec![].into(),
    };
    let init = Transaction {
        kind: TxKind::Faucet,

        inputs: vec![],
        outputs: vec![fixed_output.clone()],
        fee: 0.into(),
        data: vec![].into(),
        covenants: vec![],
        sigs: vec![],
    };
    let mut prev = init.output_coinid(0);
    let mut toret = vec![init];
    while toret.len() < n {
        let novyy = Transaction {
            kind: TxKind::Normal,
            inputs: vec![prev],
            outputs: vec![fixed_output.clone()],
            fee: 0.into(),
            data: vec![].into(),
            covenants: vec![Covenant::always_true().0.into()],
            sigs: vec![],
        };
        prev = novyy.output_coinid(0);
        toret.push(novyy);
    }
    toret
}

fn zerofee_state() -> State<MeshaCas> {
    let cfg = GenesisConfig {
        network: NetID::Testnet,
        init_coindata: CoinData {
            covhash: Address::coin_destroy(),
            value: 0.into(),
            denom: Denom::Mel,
            additional_data: vec![].into(),
        },
        stakes: Default::default(),
        init_fee_pool: 0.into(),
    };

    let meshacas = MeshaCas::new(meshanina::Mapping::open(Path::new("test.db")).unwrap());
    let mut state = cfg.realize(&novasmt::Database::new(meshacas));
    state.fee_multiplier = 0;
    state
}

static TEST_INPUT: Lazy<Vec<Transaction>> = Lazy::new(|| generate_txx(200000));

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
