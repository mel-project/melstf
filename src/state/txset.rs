use themelio_structs::{Transaction, TxHash};

#[derive(Default, Debug, Clone)]
pub struct TransactionSet {
    inner: imbl::OrdMap<TxHash, Transaction>,
}

impl FromIterator<Transaction> for TransactionSet {
    fn from_iter<T: IntoIterator<Item = Transaction>>(iter: T) -> Self {
        Self {
            inner: iter.into_iter().map(|tx| (tx.hash_nosigs(), tx)).collect(),
        }
    }
}

impl TransactionSet {
    /// Iterates over the transactions, sorted by transaction hash.
    pub fn iter(&self) -> impl Iterator<Item = &Transaction> {
        self.inner.values()
    }
    /// Iterates over the transactions hashes, sorted.
    pub fn iter_hashes(&self) -> impl Iterator<Item = TxHash> + '_ {
        self.inner.keys().copied()
    }

    /// Is this set empty?
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Adds a transaction to the set.
    pub fn insert(&mut self, txn: Transaction) {
        self.inner.insert(txn.hash_nosigs(), txn);
    }
}
