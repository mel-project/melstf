use melvm::Covenant;
use novasmt::InMemoryCas;

use criterion::{criterion_group, criterion_main, Criterion};
use melstf::{GenesisConfig, UnsealedState};
use melstructs::{Address, CoinData, Denom, NetID, Transaction, TxKind};
use once_cell::sync::Lazy;

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
            covenants: vec![Covenant::always_true().to_bytes()],
            sigs: vec![],
        };
        prev = novyy.output_coinid(0);
        toret.push(novyy);
    }
    toret
}

fn zerofee_state() -> UnsealedState<InMemoryCas> {
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
        init_fee_multiplier: 0,
    };
    cfg.realize(&novasmt::Database::new(InMemoryCas::default()))
}

static TEST_INPUT: Lazy<Vec<Transaction>> = Lazy::new(|| generate_txx(1000));

fn parallel_apply() {
    let mut init = zerofee_state();
    init.apply_tx_batch(&TEST_INPUT).unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("amdahl");
    group.sample_size(20);
    group.bench_function("gen 1000", |b| b.iter(|| generate_txx(1000)));
    // group.bench_function("sequential_apply", |b| b.iter(sequential_apply));
    group.bench_function("parallel_apply", |b| b.iter(parallel_apply));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
