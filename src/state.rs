mod applytx;
mod coins;
pub(crate) mod melmint;
pub(crate) mod melswap;
mod poolkey;

pub use crate::stake::*;
use crate::{
    smtmapping::*,
    tip_heights::{
        TIP_901_HEIGHT, TIP_906_HEIGHT, TIP_908_HEIGHT, TIP_909A_HEIGHT, TIP_909_HEIGHT,
    },
};
use crate::{state::applytx::StateHandle, tip_heights::TIP_902_HEIGHT};

use std::convert::TryInto;
use std::fmt::Debug;
use std::{collections::BTreeMap, io::Read};

use crate::state::melswap::PoolMapping;
use defmac::defmac;
use derivative::Derivative;
use novasmt::{dense::DenseMerkleTree, ContentAddrStore, Database, InMemoryCas};
use stdcode::StdcodeSerializeExt;
use tap::Pipe;
use themelio_structs::{
    Address, Block, BlockHeight, CoinData, CoinDataHeight, CoinID, CoinValue, ConsensusProof,
    Denom, Header, NetID, ProposerAction, Transaction, TxHash, STAKE_EPOCH,
};
use thiserror::Error;
use tmelcrypt::{HashVal, Hashable};

pub use poolkey::PoolKey;

pub use self::coins::CoinMapping;

#[derive(Error, Debug)]
/// A error that happens while applying a transaction to a state
pub enum StateError {
    #[error("malformed transaction")]
    MalformedTx,
    #[error("attempted to spend non-existent coin {:?}", .0)]
    NonexistentCoin(CoinID),
    #[error("unbalanced inputs and outputs")]
    UnbalancedInOut,
    #[error("insufficient fees (requires {0})")]
    InsufficientFees(CoinValue),
    #[error("referenced non-existent script {:?}", .0)]
    NonexistentScript(Address),
    #[error("does not satisfy script {:?}", .0)]
    ViolatesScript(Address),
    #[error("invalid sequential proof of work")]
    InvalidMelPoW,
    #[error("auction bid at wrong time")]
    BidWrongTime,
    #[error("block has wrong header after applying to previous block")]
    WrongHeader,
    #[error("tried to spend locked coin")]
    CoinLocked,
    #[error("duplicate transaction")]
    DuplicateTx,
}

/// World state of the Themelio blockchain
#[derive(Debug)]
pub struct State<C: ContentAddrStore> {
    pub network: NetID,

    pub height: BlockHeight,
    pub history: SmtMapping<C, BlockHeight, Header>,
    pub coins: CoinMapping<C>,
    pub transactions: BTreeMap<TxHash, Transaction>,

    pub fee_pool: CoinValue,
    pub fee_multiplier: u128,
    pub tips: CoinValue,

    pub dosc_speed: u128,
    pub pools: PoolMapping<C>,

    pub stakes: StakeMapping<C>,
}

impl<C: ContentAddrStore> Clone for State<C> {
    fn clone(&self) -> Self {
        Self {
            network: self.network,

            height: self.height,
            history: self.history.clone(),
            coins: self.coins.clone(),
            transactions: self.transactions.clone(),

            fee_pool: self.fee_pool,
            fee_multiplier: self.fee_multiplier,
            tips: self.tips,

            dosc_speed: self.dosc_speed,
            pools: self.pools.clone(),
            stakes: self.stakes.clone(),
        }
    }
}

fn read_bts(r: &mut impl Read, n: usize) -> Option<Vec<u8>> {
    let mut buf: Vec<u8> = vec![0; n];
    r.read_exact(&mut buf).ok()?;
    Some(buf)
}

impl<C: ContentAddrStore> State<C> {
    /// Returns true iff TIP 901 rule changes apply.
    pub fn tip_901(&self) -> bool {
        self.height >= TIP_901_HEIGHT
            || (self.network != NetID::Mainnet && self.network != NetID::Testnet)
    }

    /// Returns true iff TIP 902 rule changes apply.
    pub fn tip_902(&self) -> bool {
        self.height >= TIP_902_HEIGHT
            || (self.network != NetID::Mainnet && self.network != NetID::Testnet)
    }

    /// Returns true iff TIP 906 rule changes apply.
    pub fn tip_906(&self) -> bool {
        self.height >= TIP_906_HEIGHT
            || (self.network != NetID::Mainnet && self.network != NetID::Testnet)
    }

    /// Returns true iff TIP 908 rule changes apply.
    pub fn tip_908(&self) -> bool {
        self.height >= TIP_908_HEIGHT
            || (self.network != NetID::Mainnet && self.network != NetID::Testnet)
    }

    /// Returns true iff TIP 909 rule changes apply.
    pub fn tip_909(&self) -> bool {
        self.height >= TIP_909_HEIGHT
            || (self.network != NetID::Mainnet && self.network != NetID::Testnet)
    }

    /// Returns true iff TIP 909a rule changes apply.
    pub fn tip_909a(&self) -> bool {
        self.height >= TIP_909A_HEIGHT
            || (self.network != NetID::Mainnet && self.network != NetID::Testnet)
    }

    // /// Restores a state from its partial encoding in conjunction with a database. **Does not validate data and will panic; do not use on untrusted data**
    // pub fn from_partial_encoding_infallible(
    //     mut encoding: &[u8],
    //     db: &novasmt::Database<C>,
    // ) -> Self {
    //     defmac!(readu8 => u8::from_be_bytes(read_bts(&mut encoding, 1).unwrap().as_slice().try_into().unwrap()));
    //     defmac!(readu64 => u64::from_be_bytes(read_bts(&mut encoding, 8).unwrap().as_slice().try_into().unwrap()));
    //     defmac!(readu128 => u128::from_be_bytes(read_bts(&mut encoding, 16).unwrap().as_slice().try_into().unwrap()));
    //     defmac!(readtree => SmtMapping::new(db.get_tree(
    //         read_bts(&mut encoding, 32).unwrap().as_slice().try_into().unwrap(),
    //     ).unwrap()));
    //     let network: NetID = readu8!().try_into().unwrap();
    //     let height = readu64!();
    //     let history = readtree!();
    //     let coins = db
    //         .get_tree(
    //             read_bts(&mut encoding, 32)
    //                 .unwrap()
    //                 .as_slice()
    //                 .try_into()
    //                 .unwrap(),
    //         )
    //         .unwrap();

    //     let transactions: SmtMapping<C, TxHash, Transaction> = readtree!();

    //     let fee_pool = readu128!();
    //     let fee_multiplier = readu128!();
    //     let tips = readu128!();

    //     let dosc_multiplier = readu128!();
    //     let pools = readtree!();

    //     let stakes = readtree!();
    //     State {
    //         network,
    //         height: height.into(),
    //         history,
    //         coins: CoinMapping::new(coins),
    //         transactions: transactions
    //             .val_iter()
    //             .map(|t| (t.hash_nosigs(), t))
    //             .collect(),

    //         fee_pool: fee_pool.into(),
    //         fee_multiplier,
    //         tips: tips.into(),

    //         dosc_speed: dosc_multiplier,
    //         pools,

    //         stakes,
    //     }
    // }

    /// Applies a single transaction.
    pub fn apply_tx(&mut self, tx: &Transaction) -> Result<(), StateError> {
        self.apply_tx_batch(std::slice::from_ref(tx))
    }

    /// Applies a whole lot of transactions.
    pub fn apply_tx_batch(&mut self, txx: &[Transaction]) -> Result<(), StateError> {
        let old_hash = HashVal(self.coins.inner().root_hash());
        StateHandle::new(self).apply_tx_batch(txx)?.commit();
        log::debug!(
            "applied a batch of {} txx to {:?} => {:?}",
            txx.len(),
            old_hash,
            HashVal(self.coins.inner().root_hash())
        );
        Ok(())
    }

    /// Calculates the "transactions" root hash. Note that this is different depending on whether the block is pre-tip908 or post-tip908.
    pub fn transactions_root_hash(&self) -> HashVal {
        if self.tip_908() {
            let mut vv = Vec::new();
            for tx in self.transactions.values() {
                let complex = tx.hash_nosigs().pipe(|nosigs_hash| {
                    let mut v = nosigs_hash.0.to_vec();
                    v.extend_from_slice(&tx.stdcode().hash().0);
                    v
                });
                vv.push(complex);
            }
            vv.sort_unstable();
            HashVal(DenseMerkleTree::new(&vv).root_hash())
        } else {
            let db = Database::new(InMemoryCas::default());
            let mut smt = SmtMapping::new(db.get_tree(Default::default()).unwrap());
            for (k, v) in self.transactions.iter() {
                smt.insert(*k, v.clone());
            }
            smt.root_hash()
        }
    }

    /// Finalizes a state into a block. This consumes the state.
    pub fn seal(mut self, action: Option<ProposerAction>) -> SealedState<C> {
        // first apply melmint
        self = crate::melmint::preseal_melmint(self);
        assert!(self.pools.val_iter().count() >= 2);

        // then apply tip 909
        if self.tip_909() {
            let divider = self.height.0.saturating_sub(TIP_909_HEIGHT.0) / 1_000_000;
            let reward = (1u128 << 20) >> divider;
            let tip909a_erg_subsidy = reward >> 8;
            // fee subsidy
            let fee_subsidy = if self.tip_909a() {
                reward - tip909a_erg_subsidy
            } else {
                reward / 2
            };
            let mut smpool = self
                .pools
                .get(&PoolKey::new(Denom::Mel, Denom::Sym))
                .0
                .unwrap();
            let (mel, _) = smpool.swap_many(0, fee_subsidy);
            self.pools
                .insert(PoolKey::new(Denom::Mel, Denom::Sym), smpool);
            self.fee_pool += CoinValue(mel);
            // erg subsidy
            let erg_subsidy = if self.tip_909a() {
                tip909a_erg_subsidy
            } else {
                reward - fee_subsidy
            };
            let mut espool = self
                .pools
                .get(&PoolKey::new(Denom::Erg, Denom::Sym))
                .0
                .unwrap();
            let _ = espool.swap_many(0, erg_subsidy);
            self.pools
                .insert(PoolKey::new(Denom::Erg, Denom::Sym), espool);
        }

        let after_tip_901 = self.tip_901();

        // apply the proposer action
        if let Some(action) = action {
            // first let's move the fee multiplier
            let max_movement = if after_tip_901 {
                ((self.fee_multiplier >> 7) as i64).max(2)
            } else {
                (self.fee_multiplier >> 7) as i64
            };
            let scaled_movement = max_movement * action.fee_multiplier_delta as i64 / 128;
            log::debug!(
                "changing fee multiplier {} by {}",
                self.fee_multiplier,
                scaled_movement
            );
            if scaled_movement >= 0 {
                self.fee_multiplier += scaled_movement as u128;
            } else {
                self.fee_multiplier -= scaled_movement.abs() as u128;
            }

            // then it's time to collect the fees dude! we synthesize a coin with 1/65536 of the fee pool and all the tips.
            let base_fees = CoinValue(self.fee_pool.0 >> 16);
            self.fee_pool -= base_fees;
            let tips = self.tips;
            self.tips = 0.into();
            let pseudocoin_id = CoinID::proposer_reward(self.height);
            let pseudocoin_data = CoinDataHeight {
                coin_data: CoinData {
                    covhash: action.reward_dest,
                    value: base_fees + tips,
                    denom: Denom::Mel,
                    additional_data: vec![],
                },
                height: self.height,
            };
            // insert the fake coin
            self.coins
                .insert_coin(pseudocoin_id, pseudocoin_data, self.tip_906());
        }
        // create the finalized state
        SealedState(self, action)
    }
}

/// SealedState represents an immutable state at a finalized block height.
/// It cannot be constructed except through sealing a State or restoring from persistent storage.
#[derive(Derivative, Debug)]
#[derivative(Clone(bound = ""))]
pub struct SealedState<C: ContentAddrStore>(State<C>, Option<ProposerAction>);

impl<C: ContentAddrStore> SealedState<C> {
    /// Regenerate from a block, given a database to get the SMTs out of.
    pub fn from_block(blk: &Block, db: &Database<C>) -> Self {
        let coins = CoinMapping::new(db.get_tree(blk.header.coins_hash.0).unwrap());
        let history = SmtMapping::new(db.get_tree(blk.header.history_hash.0).unwrap());
        let stakes = SmtMapping::new(db.get_tree(blk.header.stakes_hash.0).unwrap());
        let pools = SmtMapping::new(db.get_tree(blk.header.pools_hash.0).unwrap());
        let state = State {
            network: blk.header.network,
            height: blk.header.height,
            history,
            coins,
            transactions: blk
                .transactions
                .iter()
                .map(|tx| (tx.hash_nosigs(), tx.clone()))
                .collect(),
            fee_pool: blk.header.fee_pool,
            fee_multiplier: blk.header.fee_multiplier,
            tips: CoinValue(0),
            dosc_speed: blk.header.dosc_speed,
            pools,
            stakes,
        };
        Self(state, blk.proposer_action)
    }

    /// From raw parts
    pub fn from_parts(state: State<C>, prop_action: Option<ProposerAction>) -> Self {
        Self(state, prop_action)
    }

    /// Returns a reference to the State finalized within.
    pub fn inner_ref(&self) -> &State<C> {
        &self.0
    }

    /// Returns whether or not it's empty.
    pub fn is_empty(&self) -> bool {
        self.1.is_none() && self.inner_ref().transactions.is_empty()
    }

    /// Returns the block header represented by the finalized state.
    pub fn header(&self) -> Header {
        let inner = &self.0;
        // panic!()
        Header {
            network: inner.network,
            previous: (inner.height.0.checked_sub(1))
                .map(|height| inner.history.get(&BlockHeight(height)).0.unwrap().hash())
                .unwrap_or_default(),
            height: inner.height,
            history_hash: inner.history.root_hash(),
            coins_hash: inner.coins.root_hash(),
            transactions_hash: inner.transactions_root_hash(),
            fee_pool: inner.fee_pool,
            fee_multiplier: inner.fee_multiplier,
            dosc_speed: inner.dosc_speed,
            pools_hash: inner.pools.root_hash(),
            stakes_hash: inner.stakes.root_hash(),
        }
    }

    /// Returns the proposer action.
    pub fn proposer_action(&self) -> Option<&ProposerAction> {
        self.1.as_ref()
    }

    /// Returns the final state represented as a "block" (header + transactions).
    pub fn to_block(&self) -> Block {
        Block {
            header: self.header(),
            transactions: self.inner_ref().transactions.values().cloned().collect(),
            proposer_action: self.1,
        }
    }
    /// Creates a new unfinalized state representing the next block.
    pub fn next_state(&self) -> State<C> {
        // static CACHE: Lazy<DashMap<HashVal, Vec<u8>>> = Lazy::new(Default::default);
        // if let Some(v) = CACHE.get(&self.header().hash()) {
        //     State::from_partial_encoding_infallible(
        //         v.value(),
        //         &self.inner_ref().coins.inner().database(),
        //     )
        // } else {
        let mut new = State::clone(self.inner_ref());
        // fee variables
        new.history.insert(self.0.height, self.header());
        new.height += BlockHeight(1);
        new.stakes.remove_stale((new.height / STAKE_EPOCH).0);
        new.transactions.clear();
        // TIP-906 transition
        if new.tip_906() && !self.inner_ref().tip_906() {
            log::warn!("DOING TIP-906 TRANSITION NOW!");
            let old_tree = new.coins.inner().clone();
            let mut count = old_tree.count();
            for (_, v) in old_tree.iter() {
                let cdh: CoinDataHeight =
                    stdcode::deserialize(&v).expect("pre-tip906 coin tree has non-cdh elements?!");
                let old_count = new.coins.coin_count(cdh.coin_data.covhash);
                new.coins
                    .insert_coin_count(cdh.coin_data.covhash, old_count + 1);
                if count % 100 == 0 {
                    log::warn!("{} left", count);
                }
                count -= 1;
            }
        }
        new
        // }
    }

    /// Applies a block to this state.
    pub fn apply_block(&self, block: &Block) -> Result<SealedState<C>, StateError> {
        let mut basis = self.next_state();
        assert!(basis.pools.val_iter().count() >= 2);
        let transactions = block.transactions.iter().cloned().collect::<Vec<_>>();
        basis.apply_tx_batch(&transactions)?;
        assert!(basis.pools.val_iter().count() >= 2);
        let basis = basis.seal(block.proposer_action);
        assert!(basis.inner_ref().pools.val_iter().count() >= 2);

        if basis.header() != block.header {
            log::warn!(
                "post-apply header {:#?} doesn't match declared header {:#?} with {} txx",
                basis.header(),
                block.header,
                transactions.len()
            );
            block.transactions.iter().for_each(|tx| {
                log::warn!("{:?}", tx);
            });

            Err(StateError::WrongHeader)
        } else {
            Ok(basis)
        }
    }

    /// Confirms a state with a given consensus proof. If called with a second argument, this function is supposed to be called to *verify* the consensus proof.
    ///
    /// **TODO**: Right now it DOES NOT check the consensus proof!
    pub fn confirm(
        self,
        cproof: ConsensusProof,
        _previous_state: Option<&State<C>>,
    ) -> Option<ConfirmedState<C>> {
        Some(ConfirmedState {
            state: self,
            cproof,
        })
    }
}

/// ConfirmedState represents a fully confirmed state with a consensus proof.
#[derive(Derivative, Debug)]
#[derivative(Clone(bound = ""))]
pub struct ConfirmedState<C: ContentAddrStore> {
    state: SealedState<C>,
    cproof: ConsensusProof,
}

impl<C: ContentAddrStore> ConfirmedState<C> {
    /// Returns the wrapped finalized state
    pub fn inner(&self) -> &SealedState<C> {
        &self.state
    }

    /// Returns the proof
    pub fn cproof(&self) -> &ConsensusProof {
        &self.cproof
    }
}

#[cfg(test)]
mod tests {
    use novasmt::{Database, InMemoryCas};
    use themelio_structs::{CoinData, CoinValue, Denom, NetID, Transaction};

    use crate::{melvm::Covenant, GenesisConfig};

    #[test]
    fn simple_dmt() {
        // let mut test_state = GenesisConfig {
        //     network: NetID::Custom02,
        //     init_coindata: CoinData {
        //         value: CoinValue(10000),
        //         denom: Denom::Mel,
        //         additional_data: vec![],
        //         covhash: Covenant::always_true().hash(),
        //     },
        //     stakes: Default::default(),
        //     init_fee_pool: CoinValue(0),
        // }
        // .realize(&Database::new(InMemoryCas::default()))
        // .seal(None)
        // .next_state();
        // // insert a bunch of transactions, then make sure all of them have valid proofs of inclusion
        // let txx_to_insert = (0..1000).map(|i| {
        //     let txn = Transaction{
        //         kind: TxKind::Faucet,
        //         inputs: vec![],
        //         outputs: vec![
        //             CoinData {
        //                 value: CoinValue(10000),
        //                 denom: Denom::Mel,
        //                 additional_data: vec![],
        //                 covhash: Covenant::always_true().hash(),
        //             },
        //             CoinData {
        //                 value: CoinValue(10000),
        //                 denom: Denom::Mel,
        //                 additional_data: vec![],
        //                 covhash: Covenant::always_true().hash(),
        //             },
        //         ],
        //     };
        // }
    }
}
