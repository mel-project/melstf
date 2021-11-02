use crate::{
    melvm::{Address, Covenant},
    CoinData, CoinID, CoinValue, Denom, Transaction, TxKind,
};

use std::collections::{BTreeMap, BTreeSet};

use tap::Pipe;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum TransactionBuildError {
    #[error("not well-formed")]
    NotWellFormed,
    #[error("inputs and outputs unbalanced")]
    Unbalanced,
    #[error("missing a covenant with hash {0}")]
    MissingCovenant(Address),
}

/// A helper struct for creating transactions.
#[derive(Debug)]
pub struct TransactionBuilder {
    in_progress: Transaction,
    required_covenants: BTreeSet<Address>,
    given_covenants: BTreeSet<Address>,
    in_balance: BTreeMap<Denom, CoinValue>,
    out_balance: BTreeMap<Denom, CoinValue>,
}

impl TransactionBuilder {
    /// Creates a new TransactionBuilder.
    pub fn new() -> Self {
        let in_progress = Transaction::empty_test();
        TransactionBuilder {
            in_progress,
            required_covenants: BTreeSet::new(),
            given_covenants: BTreeSet::new(),
            in_balance: BTreeMap::new(),
            out_balance: BTreeMap::new(),
        }
    }

    /// Sets the kind.
    pub fn kind(mut self, kind: TxKind) -> Self {
        self.in_progress.kind = kind;
        self
    }

    /// Adds an input. A CoinData must be provided.
    pub fn input(mut self, coin_id: CoinID, coin_data: CoinData) -> Self {
        self.in_progress.inputs.push(coin_id);
        *self.in_balance.entry(coin_data.denom).or_default() += coin_data.value;
        self.required_covenants.insert(coin_data.covhash);
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

    /// Adds a covenant script.
    pub fn script(mut self, script: Covenant) -> Self {
        self.given_covenants.insert(script.hash());
        self.in_progress.scripts.push(script);
        self
    }

    /// Adds a fee.
    pub fn fee(mut self, fee: CoinValue) -> Self {
        *self.out_balance.entry(Denom::Mel).or_default() += fee;
        self.in_progress.fee += fee;
        self
    }

    /// "Automatically" adds the base fee. An upper-bound for the number of signatures and the size of each signature is required.
    pub fn auto_base_fee(
        self,
        fee_multiplier: u128,
        max_sig_count: usize,
        max_sig_size: usize,
    ) -> Self {
        let fee = self.in_progress.clone().pipe(|mut tx| {
            let range = 0..max_sig_count;

            range.into_iter().for_each(|_index| {
                tx.sigs.push(vec![0; max_sig_size].into())
            });

            tx.base_fee(fee_multiplier, 0)
        });
        self.fee(fee)
    }

    /// Balance the transaction on the given denomination. Sends any excess to a change output.
    pub fn change(mut self, denom: Denom, change_addr: Address) -> Self {
        let input = self.in_balance.get(&denom).copied().unwrap_or_default();
        let output = self.out_balance.get(&denom).copied().unwrap_or_default();
        if input >= output {
            let delta = input - output;
            self = self.output(CoinData {
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
            Err(TransactionBuildError::Unbalanced)
        } else if !self.in_progress.is_well_formed() {
            Err(TransactionBuildError::NotWellFormed)
        } else {
            let was_covenant_creation_successful: Result<(), TransactionBuildError> = self.required_covenants.iter().try_for_each(|cov| {
                let is_covenant_missing_from_given_covenants: bool = !self.given_covenants.contains(cov);

                match is_covenant_missing_from_given_covenants {
                    true => Err(TransactionBuildError::MissingCovenant(*cov)),
                    false => Ok(()),
                }
            });

            match was_covenant_creation_successful {
                Ok(()) => Ok(self.in_progress),
                Err(error) => Err(error),
            }
        }
    }

    /// Sets the associated data.
    pub fn data(mut self, data: Vec<u8>) -> Self {
        self.in_progress.data = data;
        self
    }
}

impl Default for TransactionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::{melvm::Covenant, GenesisConfig};

    use super::*;

    #[test]
    fn txbuilder_basic_balance() {
        let forest = novasmt::Forest::new(novasmt::InMemoryBackend::default());
        let init_coindata = CoinData {
            denom: Denom::Mel,
            value: CoinValue::from_millions(1000u64),
            additional_data: vec![],
            covhash: Covenant::always_true().hash(),
        };
        let mut state = GenesisConfig {
            init_coindata: init_coindata.clone(),
            ..GenesisConfig::std_testnet()
        }
        .realize(&forest);
        state
            .apply_tx(
                &TransactionBuilder::new()
                    .input(CoinID::zero_zero(), init_coindata)
                    .fee(20000.into())
                    .output(CoinData {
                        covhash: Covenant::always_true().hash(),
                        value: 1000.into(),
                        denom: Denom::Mel,
                        additional_data: vec![],
                    })
                    .script(Covenant::always_true())
                    .change(Denom::Mel, Covenant::always_true().hash())
                    .build()
                    .expect("build failed"),
            )
            .expect("tx failed");
    }
}
