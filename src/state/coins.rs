use derivative::Derivative;
use novasmt::ContentAddrStore;
use stdcode::StdcodeSerializeExt;
use themelio_structs::{Address, CoinDataHeight, CoinID};
use tmelcrypt::{HashVal, Hashable};

/// A mapping that contains the coins, exposing a safeish API for the rest of the crate.
///
/// Apply
///
/// For more details, see the [Yellow Paper](https://docs.themelio.org/specifications/yellow/)
#[derive(Debug, Derivative)]
#[derivative(Clone(bound = ""))]
pub struct CoinMapping<C: ContentAddrStore> {
    inner: novasmt::Tree<C>,
}

impl<C: ContentAddrStore> CoinMapping<C> {
    /// Create a new CoinMapping.
    pub fn new(inner: novasmt::Tree<C>) -> Self {
        Self { inner }
    }

    /// Inner SMT mapping.
    pub fn inner(&self) -> &novasmt::Tree<C> {
        &self.inner
    }

    /// Root hash.
    pub fn root_hash(&self) -> HashVal {
        HashVal(self.inner.root_hash())
    }

    /// Inserts a coin into the coin mapping.
    pub fn insert_coin(&mut self, id: CoinID, data: CoinDataHeight, tip_906: bool) {
        let id = id.stdcode();
        let preexist = !self.inner.get(tmelcrypt::hash_single(&id).0).is_empty();
        self.inner
            .insert(tmelcrypt::hash_single(&id).0, &data.stdcode());
        if tip_906 && !preexist {
            let count_key = tmelcrypt::hash_keyed(b"coin_count", data.coin_data.covhash.0);
            let previous_count: u64 = self.coin_count(data.coin_data.covhash);
            self.inner
                .insert(count_key.0, &(previous_count + 1).stdcode());
        }
    }

    /// Gets a coin from the mapping.
    pub fn get_coin(&self, id: CoinID) -> Option<CoinDataHeight> {
        let bts = self.inner.get(id.stdcode().hash().0);
        if bts.is_empty() {
            None
        } else {
            Some(stdcode::deserialize(&bts).unwrap())
        }
    }

    /// Removes a coin from the coin mapping.
    pub fn remove_coin(&mut self, id: CoinID, tip_906: bool) {
        let id = id.stdcode();
        if tip_906 {
            let existing = self.inner.get(tmelcrypt::hash_single(&id).0);
            if !existing.is_empty() {
                let data: CoinDataHeight = stdcode::deserialize(&existing).unwrap();
                let count = self.coin_count(data.coin_data.covhash);
                self.insert_coin_count(data.coin_data.covhash, count - 1);
            }
        }
        self.inner.insert(tmelcrypt::hash_single(&id).0, b"");
    }

    /// Gets the coin count
    pub fn coin_count(&self, covhash: Address) -> u64 {
        let count_key = tmelcrypt::hash_keyed(b"coin_count", covhash.0);
        let v = self.inner.get(count_key.0);
        if v.is_empty() {
            0
        } else {
            stdcode::deserialize(&v).unwrap()
        }
    }

    pub fn insert_coin_count(&mut self, covhash: Address, count: u64) {
        let count_key = tmelcrypt::hash_keyed(b"coin_count", covhash.0);
        if count == 0 {
            self.inner.insert(count_key.0, b"");
        } else {
            self.inner.insert(count_key.0, &count.stdcode())
        }
    }
}
