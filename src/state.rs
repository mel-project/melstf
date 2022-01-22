mod applytx;
mod coins;
pub(crate) mod melmint;
pub(crate) mod melswap;
mod poolkey;

pub use crate::stake::*;
use crate::{
    smtmapping::*,
    tip_heights::{TIP_901_HEIGHT, TIP_906_HEIGHT, TIP_908_HEIGHT},
};
use crate::{state::applytx::StateHandle, tip_heights::TIP_902_HEIGHT};

use std::io::Read;
use std::{collections::HashSet, fmt::Debug};
use std::{convert::TryInto, sync::Arc};

use crate::state::melswap::PoolMapping;
use dashmap::lock::RwLock;
use defmac::defmac;
use derivative::Derivative;
use novasmt::{dense::DenseMerkleTree, ContentAddrStore};
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
    pub transactions: SmtMapping<C, TxHash, Transaction>,

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

    /// Generates an encoding of the state that, in conjunction with a SMT database, can recover the entire state.
    pub fn partial_encoding(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&[self.network.into()]);
        out.extend_from_slice(&self.height.0.to_be_bytes());
        out.extend_from_slice(&self.history.root_hash());
        out.extend_from_slice(&self.coins.root_hash());
        out.extend_from_slice(&self.transactions.root_hash());

        out.extend_from_slice(&self.fee_pool.0.to_be_bytes());
        out.extend_from_slice(&self.fee_multiplier.to_be_bytes());
        out.extend_from_slice(&self.tips.0.to_be_bytes());

        out.extend_from_slice(&self.dosc_speed.to_be_bytes());
        out.extend_from_slice(&self.pools.root_hash());

        out.extend_from_slice(&self.stakes.root_hash());
        out
    }

    /// Restores a state from its partial encoding in conjunction with a database. **Does not validate data and will panic; do not use on untrusted data**
    pub fn from_partial_encoding_infallible(
        mut encoding: &[u8],
        db: &novasmt::Database<C>,
    ) -> Self {
        defmac!(readu8 => u8::from_be_bytes(read_bts(&mut encoding, 1).unwrap().as_slice().try_into().unwrap()));
        defmac!(readu64 => u64::from_be_bytes(read_bts(&mut encoding, 8).unwrap().as_slice().try_into().unwrap()));
        defmac!(readu128 => u128::from_be_bytes(read_bts(&mut encoding, 16).unwrap().as_slice().try_into().unwrap()));
        defmac!(readtree => SmtMapping::new(db.get_tree(
            read_bts(&mut encoding, 32).unwrap().as_slice().try_into().unwrap(),
        ).unwrap()));
        let network: NetID = readu8!().try_into().unwrap();
        let height = readu64!();
        let history = readtree!();
        let coins = db
            .get_tree(
                read_bts(&mut encoding, 32)
                    .unwrap()
                    .as_slice()
                    .try_into()
                    .unwrap(),
            )
            .unwrap();
        let transactions = readtree!();

        let fee_pool = readu128!();
        let fee_multiplier = readu128!();
        let tips = readu128!();

        let dosc_multiplier = readu128!();
        let pools = readtree!();

        let stakes = readtree!();
        State {
            network,
            height: height.into(),
            history,
            coins: CoinMapping::new(coins),
            transactions,

            fee_pool: fee_pool.into(),
            fee_multiplier,
            tips: tips.into(),

            dosc_speed: dosc_multiplier,
            pools,

            stakes,
        }
    }

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
            for tx in self.transactions.val_iter() {
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
            self.transactions.root_hash()
        }
    }

    /// Finalizes a state into a block. This consumes the state.
    pub fn seal(mut self, action: Option<ProposerAction>) -> SealedState<C> {
        // first apply melmint
        self = crate::melmint::preseal_melmint(self);
        assert!(self.pools.val_iter().count() >= 2);

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
        SealedState(self, action, Default::default())
    }
}

/// SealedState represents an immutable state at a finalized block height.
/// It cannot be constructed except through sealing a State or restoring from persistent storage.
#[derive(Derivative, Debug)]
#[derivative(Clone(bound = ""))]
pub struct SealedState<C: ContentAddrStore>(
    State<C>,
    Option<ProposerAction>,
    Arc<RwLock<Option<State<C>>>>,
);

impl<C: ContentAddrStore> SealedState<C> {
    /// From raw parts
    pub fn from_parts(state: State<C>, prop_action: Option<ProposerAction>) -> Self {
        Self(state, prop_action, Default::default())
    }

    /// Returns a reference to the State finalized within.
    pub fn inner_ref(&self) -> &State<C> {
        &self.0
    }

    /// Returns whether or not it's empty.
    pub fn is_empty(&self) -> bool {
        self.1.is_none() && self.inner_ref().transactions.root_hash() == Default::default()
    }

    /// Returns the **partial** encoding, which must be combined with a SMT database to reconstruct the actual state.
    pub fn partial_encoding(&self) -> Vec<u8> {
        let tmp = (self.0.partial_encoding(), &self.1);
        stdcode::serialize(&tmp).unwrap()
    }

    /// Decodes from the partial encoding.
    pub fn from_partial_encoding_infallible(bts: &[u8], db: &novasmt::Database<C>) -> Self {
        let tmp: (Vec<u8>, Option<ProposerAction>) = stdcode::deserialize(bts).unwrap();
        SealedState(
            State::from_partial_encoding_infallible(&tmp.0, db),
            tmp.1,
            Default::default(),
        )
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
        let mut txx = HashSet::default();
        self.0.transactions.val_iter().for_each(|tx| {
            txx.insert(tx);
        });

        // self check since imbl sometimes is buggy
        self.0.transactions.val_iter().for_each(|tx| {
            assert!(txx.contains(&tx));
        });

        Block {
            header: self.header(),
            transactions: txx,
            proposer_action: self.1,
        }
    }
    /// Creates a new unfinalized state representing the next block.
    pub fn next_state(&self) -> State<C> {
        let next = self.2.read();
        if let Some(next) = next.clone() {
            return next;
        }
        drop(next);
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
        *self.2.write() = Some(new.clone());
        new
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
