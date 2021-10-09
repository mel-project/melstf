use crate::{melvm::Address, CoinData, CoinID, CoinValue, Denom, Transaction, TxKind};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum TransactionBuildError {
    #[error("not well-formed")]
    NotWellFormed,
    #[error("inputs and outputs unbalanced")]
    Unbalanced,
}

/// A helper struct for creating transactions.
#[derive(Debug)]
pub struct TransactionBuilder {
    in_progress: Transaction,
    in_balance: BTreeMap<Denom, CoinValue>,
    out_balance: BTreeMap<Denom, CoinValue>,
}

impl TransactionBuilder {
    /// Creates a new TransactionBuilder.
    pub fn new() -> Self {
        let in_progress = Transaction::empty_test();
        TransactionBuilder {
            in_progress,
            in_balance: BTreeMap::new(),
            out_balance: BTreeMap::new(),
        }
    }

    /// Sets the kind.
    pub fn kind(mut self, kind: TxKind) -> Self {
        self.in_progress.kind = kind;
        self
    }

    /// Adds an input. A denomination and value must be provided.
    pub fn input(mut self, coin_id: CoinID, denom: Denom, value: CoinValue) -> Self {
        self.in_progress.inputs.push(coin_id);
        *self.in_balance.entry(denom).or_default() += value;
        self
    }

    /// Adds an output.
    pub fn output(mut self, data: CoinData) -> Self {
        if data.denom != Denom::NewCoin {
            *self.out_balance.entry(data.denom).or_default() += data.value;
        }
        self.in_progress.outputs.push(data);
        self
    }

    /// Adds a fee.
    pub fn fee(mut self, fee: CoinValue) -> Self {
        *self.out_balance.entry(Denom::Mel).or_default() += fee;
        self.in_progress.fee += fee;
        self
    }

    /// Balance the transaction on the given denomination. Sends any excess to a change output.
    pub fn balance(mut self, denom: Denom, change_addr: Address) -> Self {
        let input = self.in_balance.get(&denom).copied().unwrap_or_default();
        let output = self.out_balance.get(&denom).copied().unwrap_or_default();
        if input >= output {
            let delta = input - output;
            self.in_progress.outputs.push(CoinData {
                covhash: change_addr,
                value: delta,
                denom,
                additional_data: vec![],
            });
        }
        self
    }

    /// Attempts to generate the transaction.
    pub fn build(self) -> Result<Transaction, TransactionBuildError> {
        if self.in_balance != self.out_balance {
            return Err(TransactionBuildError::Unbalanced);
        }
        if !self.in_progress.is_well_formed() {
            return Err(TransactionBuildError::NotWellFormed);
        }
        Ok(self.in_progress)
    }

    /// Sets the associated data.
    pub fn data(mut self, data: Vec<u8>) -> Self {
        self.in_progress.data = data;
        self
    }
}
