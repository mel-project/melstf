use std::collections::BinaryHeap;

// use crate::testing::factory::TransactionFactory;
// use crate::testing::fixtures::SEND_MEL_AMOUNT;
use crate::{CoinData, CoinID, CoinValue, Denom, melvm, Transaction, TxKind};

pub fn random_valid_txx(
    rng: &mut impl rand::Rng,
    start_coin: CoinID,
    start_coindata: CoinData,
    signer: tmelcrypt::Ed25519SK,
    covenant: &melvm::Covenant,
    fee: CoinValue,
) -> Vec<Transaction> {
    random_valid_txx_count(rng, start_coin, start_coindata, signer, covenant, fee, 100)
}

pub fn random_valid_txx_count(
    rng: &mut impl rand::Rng,
    start_coin: CoinID,
    start_coindata: CoinData,
    signer: tmelcrypt::Ed25519SK,
    covenant: &melvm::Covenant,
    fee: CoinValue,
    tx_count: u32,
) -> Vec<Transaction> {
    let mut pqueue: BinaryHeap<(u64, CoinID, CoinData)> = BinaryHeap::new();
    pqueue.push((rng.gen(), start_coin, start_coindata));
    let mut toret = Vec::new();
    for _ in 0..tx_count {
        // pop one item from pqueue
        let (_, to_spend, to_spend_data) = pqueue.pop().unwrap();
        assert_eq!(to_spend_data.covhash, covenant.hash());
        let mut new_tx = Transaction {
            kind: TxKind::Normal,
            inputs: vec![to_spend],
            outputs: vec![CoinData {
                covhash: covenant.hash(),
                value: to_spend_data.value - fee,
                denom: Denom::Mel,
                additional_data: vec![],
            }],
            fee,
            scripts: vec![covenant.clone()],
            data: vec![],
            sigs: vec![],
        };
        new_tx = new_tx.signed_ed25519(signer);
        for (i, out) in new_tx.outputs.iter().enumerate() {
            let cin = CoinID {
                txhash: new_tx.hash_nosigs(),
                index: i as u8,
            };
            pqueue.push((rng.gen(), cin, out.clone()));
        }
        toret.push(new_tx);
    }
    toret
}

// pub fn fee_estimate() -> CoinValue {
//     // Assuming some fee for tx (use higher multiplier to ensure its enough)
//     let fee_multiplier = 10000;
//     let fee = TransactionFactory::new()
//         .build(|_| {})
//         .weight()
//         .saturating_mul(fee_multiplier);
//     fee.into()
// }