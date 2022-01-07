mod applytx;
pub(crate) mod melmint;
pub(crate) mod melswap;
mod poolkey;

pub use crate::stake::*;
use crate::state::applytx::StateHandle;
use crate::{constants::*, melvm::Address, preseal_melmint, CoinDataHeight, Denom, TxHash};
use crate::{smtmapping::*, BlockHeight, CoinData, CoinValue};
use crate::{transaction::Transaction, CoinID};

use std::{collections::BTreeMap, convert::TryInto};
use std::{collections::BTreeSet, io::Read};
use std::{collections::HashSet, fmt::Debug};

use arbitrary::Arbitrary;
use defmac::defmac;
use derivative::Derivative;
use novasmt::ContentAddrStore;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};
use thiserror::Error;
use tmelcrypt::{Ed25519PK, HashVal};

use crate::state::melswap::PoolMapping;

pub use poolkey::PoolKey;

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

/// Identifies a network.
#[derive(
    Clone,
    Copy,
    IntoPrimitive,
    TryFromPrimitive,
    Eq,
    PartialEq,
    Debug,
    Serialize_repr,
    Deserialize_repr,
    Hash,
    Arbitrary,
)]
#[repr(u8)]
pub enum NetID {
    Testnet = 0x01,
    Custom02 = 0x02,
    Custom03 = 0x03,
    Custom04 = 0x04,
    Custom05 = 0x05,
    Custom06 = 0x06,
    Custom07 = 0x07,
    Custom08 = 0x08,
    Mainnet = 0xff,
}

/// World state of the Themelio blockchain
#[derive(Debug)]
pub struct State<C: ContentAddrStore> {
    pub network: NetID,

    pub height: BlockHeight,
    pub history: SmtMapping<C, BlockHeight, Header>,
    pub coins: SmtMapping<C, CoinID, CoinDataHeight>,
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

            height: self.height.clone(),
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
        let coins = readtree!();
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
            coins,
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

    pub fn apply_tx_batch(&mut self, txx: &[Transaction]) -> Result<(), StateError> {
        let old_hash = self.coins.root_hash();
        StateHandle::new(self).apply_tx_batch(txx)?.commit();
        log::debug!(
            "applied a batch of {} txx to {:?} => {:?}",
            txx.len(),
            old_hash,
            self.coins.root_hash()
        );
        Ok(())
    }

    /// Finalizes a state into a block. This consumes the state.
    pub fn seal(mut self, action: Option<ProposerAction>) -> SealedState<C> {
        // first apply melmint
        self = preseal_melmint(self);
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
            self.coins.insert(pseudocoin_id, pseudocoin_data);
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
        SealedState(State::from_partial_encoding_infallible(&tmp.0, db), tmp.1)
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
            transactions_hash: inner.transactions.root_hash(),
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
        let mut new = State::clone(self.inner_ref());
        // fee variables
        new.history.insert(self.0.height, self.header());
        new.height += BlockHeight(1);
        new.stakes.remove_stale((new.height / STAKE_EPOCH).0);
        new.transactions.clear();
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

/// ProposerAction describes the standard action that the proposer takes when proposing a block.
#[derive(Derivative, Serialize, Deserialize, Copy, Clone, Debug, Eq, PartialEq)]
pub struct ProposerAction {
    /// Change in fee. This is scaled to the proper size.
    pub fee_multiplier_delta: i8,
    /// Where to sweep fees.
    pub reward_dest: Address,
}

pub type ConsensusProof = BTreeMap<Ed25519PK, Vec<u8>>;

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

// impl Deref<Target =
#[derive(Serialize, Deserialize, Copy, Clone, Debug, Eq, PartialEq, Hash, Arbitrary)]
/// A block header, which commits to a particular SealedState.
pub struct Header {
    pub network: NetID,
    pub previous: HashVal,
    pub height: BlockHeight,
    pub history_hash: HashVal,
    pub coins_hash: HashVal,
    pub transactions_hash: HashVal,
    pub fee_pool: CoinValue,
    pub fee_multiplier: u128,
    pub dosc_speed: u128,
    pub pools_hash: HashVal,
    pub stakes_hash: HashVal,
}

impl Header {
    pub fn hash(&self) -> tmelcrypt::HashVal {
        tmelcrypt::hash_single(&stdcode::serialize(self).unwrap())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
/// A (serialized) block.
pub struct Block {
    pub header: Header,
    pub transactions: HashSet<Transaction>,
    pub proposer_action: Option<ProposerAction>,
}

impl Block {
    /// Abbreviate a block
    pub fn abbreviate(&self) -> AbbrBlock {
        AbbrBlock {
            header: self.header,
            txhashes: self.transactions.iter().map(|v| v.hash_nosigs()).collect(),
            proposer_action: self.proposer_action,
        }
    }
}

/// An abbreviated block
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AbbrBlock {
    pub header: Header,
    pub txhashes: BTreeSet<TxHash>,
    pub proposer_action: Option<ProposerAction>,
}
