use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::PathBuf,
};

use bytes::Bytes;
use melstructs::{
    Address, CoinData, CoinValue, Denom, Transaction, TxKind,
};

fn read_from_file(path: &PathBuf) -> HashMap<Address, HashMap<Denom, CoinValue>> {
    let file = File::open(path).expect(&format!("unable to open balances file {}", path.display()));
    let balances = serde_json::from_reader(file).expect(&format!("error decoding balances json file {}", path.display()));
    balances
}

pub fn to_faucet_transactions(
    balances: HashMap<Address, HashMap<Denom, CoinValue>>,
) -> HashSet<Transaction> {
    let mut faucet_txs = HashSet::new();
    let mut tx = Transaction::new(TxKind::Faucet);

    for (covhash, denom_to_value) in balances {
        for (denom, value) in denom_to_value {
            if tx.outputs.len() == 255 {
                faucet_txs.insert(tx.clone());
                tx.outputs.clear();
            }

            let coin_data = CoinData {
                covhash,
                value,
                denom,
                additional_data: Bytes::new(),
            };
            tx.outputs.push(coin_data);
        }
    }

    if tx.outputs.len() > 0 {
        faucet_txs.insert(tx);
    }

    faucet_txs
}


pub fn faucet_txs(path: PathBuf) -> Vec<Transaction> {
    let balances = read_from_file(&path);
    to_faucet_transactions(balances).into_iter().collect()
}
