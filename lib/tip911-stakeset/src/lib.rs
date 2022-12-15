use novasmt::{dense::DenseMerkleTree, ContentAddrStore, Database, InMemoryCas};
use stdcode::StdcodeSerializeExt;
use themelio_structs::{CoinID, CoinValue, StakeDoc, TxHash};
use tmelcrypt::{Ed25519PK, HashVal, Hashable};

/// Keeps track of all the active stakes in the blockchain. Abstracts over the pre-TIP911 and post-TIP911 representations.
pub struct StakeSet {
    stakes: imbl::HashMap<TxHash, StakeDoc>,
}

impl StakeSet {
    /// Adds another stake to the set. The transaction hash of the staking transaction, as well as the parsed [StakeDoc], must be given.
    pub fn add_stake(&mut self, txhash: TxHash, stake: StakeDoc) {
        self.stakes.insert(txhash, stake);
    }

    /// Checks whether a particular CoinID is frozen due to it containing an active stake.
    pub fn is_frozen(&self, coin: CoinID) -> bool {
        self.stakes.contains_key(&coin.txhash) && coin.index == 0
    }

    /// Checks how many votes a particular staker has, in the given epoch.
    pub fn votes(&self, epoch: u64, key: Ed25519PK) -> u128 {
        self.stakes
            .values()
            .filter(|v| v.e_start <= epoch && v.e_post_end > epoch && v.pubkey == key)
            .map(|v| v.syms_staked.0)
            .sum()
    }

    /// Obtains the number of votes in total for the given epoch.
    pub fn total_votes(&self, epoch: u64) -> u128 {
        self.stakes
            .values()
            .filter(|v| v.e_start <= epoch && v.e_post_end > epoch)
            .map(|v| v.syms_staked.0)
            .sum()
    }

    /// Removes all the stakes that have expired by this epoch.
    pub fn unlock_old(&mut self, epoch: u64) {
        self.stakes.retain(|_, v| v.e_post_end >= epoch);
    }

    /// Obtain an old-style (pre-TIP-911) sparse merkle tree that maps the **hash of the stdcode encoding of** the transaction hash of the transaction that created the stake, to the stdcode-encoded stake.
    pub fn pre_tip911(&self) -> novasmt::Tree<InMemoryCas> {
        let mut tree = Database::new(InMemoryCas::default())
            .get_tree([0u8; 32])
            .unwrap();
        for (k, v) in self.stakes.iter() {
            tree.insert(k.stdcode().hash().0, &v.stdcode());
        }
        tree
    }

    /// Obtain a TIP-911-style object.
    pub fn post_tip911(&self, epoch: u64) -> Tip911 {
        let mut stakes: Vec<(TxHash, StakeDoc)> = self.stakes.clone().into_iter().collect();
        // sort by txhash
        stakes.sort_unstable_by_key(|s| s.0);
        // then, sort *stably* by stake size.
        stakes.sort_by_key(|s| s.1.syms_staked);
        Tip911 {
            current_total: self.total_votes(epoch).into(),
            next_total: self.total_votes(epoch + 1).into(),
            stakes,
        }
    }
}

pub struct Tip911 {
    /// Tally of all the SYM that can vote in this epoch.
    pub current_total: CoinValue,
    /// Tally of all the SYM that can vote in the next epoch.
    pub next_total: CoinValue,
    /// Vector of all the stakedocs, sorted first by stake size, and then by txhash.
    pub stakes: Vec<(TxHash, StakeDoc)>,
}

impl Tip911 {
    /// Calculates a dense merkle tree where the kth element is the hash of a stdcode'ed vector of the first k(current_total, next_total, txhash, stakedoc).
    pub fn to_datablocks(&self) -> Vec<Vec<u8>> {
        let vec = self
            .stakes
            .iter()
            .map(|(txhash, sdoc)| (self.current_total, self.next_total, txhash, sdoc))
            .collect::<Vec<_>>();
        let upto_vec: Vec<Vec<u8>> = (0..vec.len())
            .map(|k| vec[..k].to_vec().stdcode())
            .collect();

        upto_vec
    }

    pub fn to_tree(&self) -> DenseMerkleTree {
        let leaves: Vec<HashVal> = self
            .to_datablocks()
            .iter()
            .map(|datablock| datablock.hash())
            .collect();

        DenseMerkleTree::new(&leaves)
    }
}
