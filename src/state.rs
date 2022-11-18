mod applytx;
mod coins;
pub(crate) mod melmint;
pub(crate) mod txset;

pub use crate::stake::*;
use crate::tip_heights::TIP_902_HEIGHT;
use crate::{
    smtmapping::*,
    state::applytx::apply_tx_batch_impl,
    tip_heights::{
        TIP_901_HEIGHT, TIP_906_HEIGHT, TIP_908_HEIGHT, TIP_909A_HEIGHT, TIP_909_HEIGHT,
    },
};

use std::fmt::Debug;

use crate::state::melmint::PoolMapping;
use derivative::Derivative;
use novasmt::{dense::DenseMerkleTree, ContentAddrStore, Database, InMemoryCas};
use stdcode::StdcodeSerializeExt;
use tap::Pipe;
use themelio_structs::{
    Address, Block, BlockHeight, CoinData, CoinDataHeight, CoinID, CoinValue, ConsensusProof,
    Denom, Header, NetID, PoolKey, PoolState, ProposerAction, StakeDoc, Transaction, TxHash,
    STAKE_EPOCH,
};
use thiserror::Error;
use tmelcrypt::{HashVal, Hashable};

pub use self::coins::CoinMapping;
use self::txset::TransactionSet;

#[derive(Error, Debug, PartialEq, Eq)]
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
    #[error("block has wrong header after applying to previous block")]
    WrongHeader,
    #[error("tried to spend locked coin")]
    CoinLocked,
    #[error("duplicate transaction")]
    DuplicateTx,
}

/// An "unsealed" world state of the Themelio blockchain.
#[derive(Debug, Derivative)]
#[derivative(Clone(bound = ""))]
pub struct State<C: ContentAddrStore> {
    /// The associated network ID for this state. The genesis state has a hardcoded network ID, and can never change afterwards.
    pub(crate) network: NetID,

    /// The number of blocks created since the genesis block. This is a type alias for a `u64`.
    pub(crate) height: BlockHeight,

    /// A sparse Merkle tree that maps every previous height with a `Header`.
    pub(crate) history: SmtMapping<C, BlockHeight, Header>,

    /// A sparse Merkle tree that maps a `CoinID` to its associated `CoinDataHeight`, which encapsulates a transaction's associated data, such as the value, denomination (MEL, SYM, etc.), and the `Covenent` constraint hash for the coin.
    pub(crate) coins: CoinMapping<C>,

    /// This contains all of the transactions within the last block.
    pub(crate) transactions: TransactionSet,

    /// The accumulated base fees that funds staker rewards. This "belongs" to all stakers.
    pub(crate) fee_pool: CoinValue,

    /// The multiper that scales the amount of fees transactions are required to have.
    pub(crate) fee_multiplier: u128,

    /// Fees local to this block that are paid to the block proposer when this `State` is sealed.
    pub(crate) tips: CoinValue,

    /// The DOSC speed that measures how much work the fastest processor can do in 24 hours. More details about this mechanism can be found in the formal [specification](https://docs.themelio.org/specifications/tech-melmint/).
    pub(crate) dosc_speed: u128,

    /// A sparse Merkle tree that maps token denominations to internal pool states.
    pub(crate) pools: PoolMapping<C>,

    /// A sparse Merkle tree that maps transaction hashes to a `StakeDoc`, which contains information on how to verify consensus proofs.
    pub(crate) stakes: StakeMapping<C>,
}

impl<C: ContentAddrStore> State<C> {
    fn tip_condition(&self, activation: BlockHeight) -> bool {
        if self.network == NetID::Mainnet {
            // incomplete things are always
            self.height >= activation
        } else if self.network == NetID::Testnet {
            self.height >= BlockHeight(27501)
        } else {
            true
        }
    }

    /// Returns true iff TIP 901 rule changes apply.
    pub(crate) fn tip_901(&self) -> bool {
        self.tip_condition(TIP_901_HEIGHT)
    }

    /// Returns true iff TIP 902 rule changes apply.
    pub(crate) fn tip_902(&self) -> bool {
        self.tip_condition(TIP_902_HEIGHT)
    }

    /// Returns true iff TIP 906 rule changes apply.
    pub(crate) fn tip_906(&self) -> bool {
        self.tip_condition(TIP_906_HEIGHT)
    }

    /// Returns true iff TIP 908 rule changes apply.
    pub(crate) fn tip_908(&self) -> bool {
        self.tip_condition(TIP_908_HEIGHT) || self.network == NetID::Custom08
    }

    /// Returns true iff TIP 909 rule changes apply.
    pub(crate) fn tip_909(&self) -> bool {
        self.tip_condition(TIP_909_HEIGHT)
    }

    /// Returns true iff TIP 909a rule changes apply.
    pub(crate) fn tip_909a(&self) -> bool {
        self.tip_condition(TIP_909A_HEIGHT)
    }

    /// Applies a single transaction.
    /// Internally, this calls [apply_tx_batch](Self::apply_tx_batch) on a slice with a length of 1.
    pub fn apply_tx(&mut self, tx: &Transaction) -> Result<(), StateError> {
        self.apply_tx_batch(std::slice::from_ref(tx))
    }

    /// Applies a whole lot of transactions.
    /// At a high level, this does the following:
    /// - Loads the relevant coins from the transaction inputs and outputs
    /// - Validates the transactions against the given coins and stakes
    /// - Validates any DOSC mint transactions
    /// - Applies any new stakes
    /// - Creates a new state from the incoming transactions
    pub fn apply_tx_batch(&mut self, txx: &[Transaction]) -> Result<(), StateError> {
        let old_hash = HashVal(self.coins.inner().root_hash());
        let new_state = apply_tx_batch_impl(self, txx)?;
        log::debug!(
            "applied a batch of {} txx to {:?} => {:?}",
            txx.len(),
            old_hash,
            HashVal(new_state.coins.inner().root_hash())
        );
        *self = new_state;
        Ok(())
    }

    /// Calculates the "transactions" root hash. Note that this is different depending on whether the block is pre-TIP-908 or post-TIP-908.
    pub fn transactions_root_hash(&self) -> HashVal {
        if self.tip_908() {
            HashVal(self.tip908_transactions().root_hash())
        } else {
            let db = Database::new(InMemoryCas::default());
            let mut smt = SmtMapping::new(db.get_tree(Default::default()).unwrap());
            for txn in self.transactions.iter() {
                smt.insert(txn.hash_nosigs(), txn.clone());
            }
            smt.root_hash()
        }
    }

    /// Obtains the dense merkle tree (TIP-908)
    pub fn tip908_transactions(&self) -> DenseMerkleTree {
        let mut vv = Vec::new();
        for tx in self.transactions.iter() {
            let complex = tx.hash_nosigs().pipe(|nosigs_hash| {
                let mut v = nosigs_hash.0.to_vec();
                v.extend_from_slice(&tx.stdcode().hash().0);
                v
            });
            vv.push(complex);
        }
        vv.sort_unstable();
        DenseMerkleTree::new(&vv)
    }

    /// Obtains the sorted position of the given transaction within this state.
    pub fn transaction_sorted_posn(&self, txhash: TxHash) -> Option<usize> {
        self.transactions
            .iter_hashes()
            .enumerate()
            .find(|(_, k)| k == &txhash)
            .map(|(i, _)| i)
    }

    /// Helper function that returns all the StakeDocs for a particular height, given the stakes in this state.
    pub fn stake_docs_for_height(
        &self,
        height: BlockHeight,
    ) -> impl Iterator<Item = StakeDoc> + '_ {
        self.stakes
            .val_iter()
            .filter(move |sd| height.epoch() >= sd.e_start && height.epoch() < sd.e_post_end)
    }

    fn apply_tip_909(&mut self) {
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
            .unwrap();
        let _ = espool.swap_many(0, erg_subsidy);
        self.pools
            .insert(PoolKey::new(Denom::Erg, Denom::Sym), espool);
    }

    fn move_action_fee_multiplier(&mut self, after_tip_901: bool, action: ProposerAction) {
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
            self.fee_multiplier -= scaled_movement.unsigned_abs() as u128;
        }
    }

    fn collect_proposer_action_fee(&mut self, action: ProposerAction) {
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
                additional_data: Default::default(),
            },
            height: self.height,
        };
        // insert the fake coin
        self.coins
            .insert_coin(pseudocoin_id, pseudocoin_data, self.tip_906());
    }

    fn apply_proposer_action(&mut self, action: ProposerAction, after_tip_901: bool) {
        // first let's move the fee multiplier
        self.move_action_fee_multiplier(after_tip_901, action);

        // collect the fees due! We synthesize a coin with 1/65536 of the fee pool and all the tips.
        self.collect_proposer_action_fee(action);
    }

    /// Finalizes a state into a block. This consumes the state.
    /// This does [some pre-processing](crate::melmint::preseal_melmint) and applies the given propose action and creates the new `SealedState`.
    /// This also means that no more transactions can be applied to this state at the current `BlockHeight`.
    pub fn seal(mut self, action: Option<ProposerAction>) -> SealedState<C> {
        // first apply melmint
        self = crate::melmint::preseal_melmint(self);
        assert!(self.pools.val_iter().count() >= 2);

        // then apply tip 909
        if self.tip_909() {
            self.apply_tip_909();
        }

        // apply the proposer action
        if let Some(action) = action {
            self.apply_proposer_action(action, self.tip_901());
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
    /// Obtains an unspent coin in the state.
    pub fn get_coin(&self, id: CoinID) -> Option<CoinDataHeight> {
        self.0.coins.get_coin(id)
    }

    /// Obtains the raw sparse Merkle tree containing the coin mapping.
    pub fn raw_coins_smt(&self) -> novasmt::Tree<C> {
        self.0.coins.inner().clone()
    }

    /// Obtains a past header from the state.
    pub fn get_history(&self, height: BlockHeight) -> Option<Header> {
        self.0.history.get(&height)
    }

    /// Obtains the raw sparse Merkle tree containing the history of block headers.
    pub fn raw_history_smt(&self) -> novasmt::Tree<C> {
        self.0.history.mapping.clone()
    }

    /// Obtains information about a particular Melswap pool.
    pub fn get_pool(&self, key: PoolKey) -> Option<PoolState> {
        self.0.pools.get(&key)
    }

    /// Obtains the raw sparse Merkle tree containing all the pools.
    pub fn raw_pools_smt(&self) -> novasmt::Tree<C> {
        self.0.pools.mapping.clone()
    }

    /// Obtains information about a particular stake, identified by the txhash of the transaction that locked up the SYM.
    pub fn get_stake(&self, key: TxHash) -> Option<StakeDoc> {
        self.0.stakes.get(&key)
    }

    /// Obtains the raw sparse Merkle tree containing all the stakes.
    pub fn raw_stakes_smt(&self) -> novasmt::Tree<C> {
        self.0.stakes.mapping.clone()
    }

    /// Regenerate from a block, given a database to get the SMTs out of.
    pub fn from_block(blk: &Block, db: &Database<C>) -> Self {
        let coins = CoinMapping::new(db.get_tree(blk.header.coins_hash.0).unwrap());
        let history = SmtMapping::new(db.get_tree(blk.header.history_hash.0).unwrap());
        let stakes = SmtMapping::new(db.get_tree(blk.header.stakes_hash.0).unwrap());
        let pools = SmtMapping::new(db.get_tree(blk.header.pools_hash.0).unwrap());
        let transactions = blk.transactions.iter().cloned().collect();
        let state = State {
            network: blk.header.network,
            height: blk.header.height,
            history,
            coins,
            transactions,
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

    /// Returns whether or not it's empty.
    pub fn is_empty(&self) -> bool {
        self.1.is_none() && self.0.transactions.is_empty()
    }

    /// Returns the block header represented by the finalized state.
    pub fn header(&self) -> Header {
        let inner = &self.0;
        Header {
            network: inner.network,
            previous: (inner.height.0.checked_sub(1))
                .map(|height| inner.history.get(&BlockHeight(height)).unwrap().hash())
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
            transactions: self.0.transactions.iter().cloned().collect(),
            proposer_action: self.1,
        }
    }

    fn apply_tip_906_for_next_state(next_state: &mut State<C>) {
        log::warn!("DOING TIP-906 TRANSITION NOW!");
        let old_tree = next_state.coins.inner().clone();
        let mut count = old_tree.count();
        for (_, v) in old_tree.iter() {
            let cdh: CoinDataHeight =
                stdcode::deserialize(&v).expect("pre-tip906 coin tree has non-cdh elements?!");
            let old_count = next_state.coins.coin_count(cdh.coin_data.covhash);
            next_state
                .coins
                .insert_coin_count(cdh.coin_data.covhash, old_count + 1);
            if count % 100 == 0 {
                log::warn!("{} left", count);
            }
            count -= 1;
        }
    }

    /// Creates a new unfinalized state representing the next block.
    ///
    /// This is the "main" operation on a `SealedState` used to advance it to a `State` that can start accepting further transactions for the next block.
    pub fn next_state(&self) -> State<C> {
        let mut new = self.0.clone();
        // fee variables
        new.history.insert(self.0.height, self.header());
        new.height += BlockHeight(1);
        new.stakes.remove_stale((new.height / STAKE_EPOCH).0);
        new.transactions = Default::default();

        // TIP-906 transition
        if new.tip_906() && !self.0.tip_906() {
            Self::apply_tip_906_for_next_state(&mut new);
        }

        new
    }

    /// Applies a block to this state.
    ///
    /// ## Usage in Full Nodes
    ///
    /// This functionality is used by full nodes (both Auditors and Stakers) to sync and move the state forward by applying new transactions.
    ///
    /// For example, when [Auditor nodes](https://github.com/themeliolabs/themelio-node/blob/master/src/protocols/node.rs) sync with other nodes, they will apply a stream of blocks to persistent storage.
    /// Every epoch loop, [Staker nodes](https://github.com/themeliolabs/themelio-node/blob/master/src/storage/storage.rs) will call `apply_block` on the latest confirmed block from the consensus algorithm.
    pub fn apply_block(&self, block: &Block) -> Result<SealedState<C>, StateError> {
        let mut basis = self.next_state();
        assert!(basis.pools.val_iter().count() >= 2);
        let transactions = block.transactions.iter().cloned().collect::<Vec<_>>();
        basis.apply_tx_batch(&transactions)?;
        assert!(basis.pools.val_iter().count() >= 2);
        let basis = basis.seal(block.proposer_action);
        assert!(basis.0.pools.val_iter().count() >= 2);

        if basis.header() != block.header {
            log::warn!(
                "post-apply header {:#?} doesn't match declared header {:#?} with {} txx",
                basis.header(),
                block.header,
                transactions.len()
            );
            block.transactions.iter().for_each(|tx| {
                log::warn!("{:?}", tx.kind);
            });

            log::warn!("pre-apply header: {:#?}", self.header());
            log::warn!("block: {:#?}", block);

            Err(StateError::WrongHeader)
        } else {
            Ok(basis)
        }
    }

    /// Confirms a state with a given consensus proof. If the proof is not valid, returns an error.
    ///
    /// **Note**: The consensus proof is checked against the stakers contained in the state itself. This seems at first glance to be obviously incorrect --- how can we trust the stakers listed by an unconfirmed state?
    ///
    /// But that is actually fine. When we want to trust a state at a given height, we're interested in two things:
    /// - Is its a **valid** state? That is, did it come from valid applications of `apply_block` starting from the genesis state?
    /// - Is it **confirmed** by signatures from enough stakers?
    ///
    /// By the time we end up calling this, we already know that we have a valid state, since otherwise we couldn't have constructed this SealedState successfully. And when we check the second one, it's okay to use the stakers in the state itself, because if it were actually valid, the voting power of the stakers must be correct in the current state.
    ///
    /// This is because there's no transaction that can change the correct voting power distribution of stakers *within the same block*. Thus, the list of stakers that applies to this state here must be the same as the previous, trusted state that we created this state from.
    pub fn confirm(&self, cproof: ConsensusProof) -> Option<ConfirmedState<C>> {
        // first check all the signatures
        for (k, sig) in cproof.iter() {
            if !k.verify(&self.header().hash(), sig) {
                return None;
            }
        }
        // then check that we have enough signatures
        let my_epoch = self.0.height.epoch();
        let relevant_stakes = self
            .0
            .stakes
            .val_iter()
            .filter(|doc| doc.e_start <= my_epoch && doc.e_post_end > my_epoch)
            .collect::<Vec<_>>();
        let total_votes: u128 = relevant_stakes.iter().map(|doc| doc.syms_staked.0).sum();
        let present_votes: u128 = relevant_stakes
            .iter()
            .filter(|doc| cproof.contains_key(&doc.pubkey))
            .map(|doc| doc.syms_staked.0)
            .sum();
        if total_votes > present_votes / 2 * 3 {
            Some(ConfirmedState {
                state: self.clone(),
                cproof,
            })
        } else {
            None
        }
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
    use std::collections::HashMap;

    use novasmt::InMemoryCas;
    use rand::prelude::SliceRandom;
    use rand::RngCore;
    use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
    use stdcode::StdcodeSerializeExt;
    use tap::Tap;
    use themelio_structs::{
        Address, CoinData, CoinID, CoinValue, Denom, NetID, StakeDoc, Transaction,
        TransactionBuilder, TxKind, MAX_COINVAL,
    };
    use tmelcrypt::Hashable;

    use crate::{
        melvm::opcode::OpCode,
        melvm::Covenant,
        testing::functions::{create_state, valid_txx},
        State, StateError,
        StateError::{
            InsufficientFees, MalformedTx, NonexistentCoin, NonexistentScript, UnbalancedInOut,
            ViolatesScript,
        },
    };

    #[test]
    fn apply_batch_exceed_maximum_coin_value() {
        let state: State<InMemoryCas> = create_state(&HashMap::new(), 0);

        let maximum_coin_value_exceeded: CoinValue =
            themelio_structs::MAX_COINVAL + themelio_structs::CoinValue(1);

        let mut transactions: Vec<Transaction> = valid_txx(tmelcrypt::ed25519_keygen());

        transactions.par_iter_mut().for_each(|transaction| {
            transaction.outputs.par_iter_mut().for_each(|coin_data| {
                coin_data.value = maximum_coin_value_exceeded;
            });
        });

        let state_error_result: Result<(), StateError> =
            state.clone().apply_tx_batch(&transactions);

        assert_eq!(state_error_result, Err(MalformedTx));
    }

    #[test]
    fn script_violation() {
        let mut state: State<InMemoryCas> = create_state(&HashMap::new(), 0);

        let (public_key, _secret_key): (tmelcrypt::Ed25519PK, tmelcrypt::Ed25519SK) =
            tmelcrypt::ed25519_keygen();

        let covenant_hash: themelio_structs::Address =
            Covenant::std_ed25519_pk_legacy(public_key).hash();

        let first_transaction: Transaction = Transaction {
            kind: TxKind::Faucet,
            inputs: vec![],
            outputs: vec![CoinData {
                covhash: covenant_hash,
                value: 20000.into(),
                denom: Denom::Mel,
                additional_data: vec![].into(),
            }],
            data: vec![].into(),
            fee: CoinValue(20000),
            covenants: vec![],
            sigs: vec![],
        };

        state.apply_tx(&first_transaction).unwrap();

        let mut covenant: Covenant = Covenant::from_ops(&[
            OpCode::PushI(1_u8.into()),
            OpCode::PushI(2_u8.into()),
            OpCode::Add,
            OpCode::PushI(3_u8.into()),
            OpCode::Eql,
        ])
        .expect("Failed to create a Add covenant.");

        covenant.0 = vec![1];

        let second_transaction: Transaction = Transaction {
            kind: TxKind::Normal,
            inputs: vec![first_transaction.output_coinid(0)],
            outputs: vec![],
            data: vec![].into(),
            fee: CoinValue(1000),
            covenants: vec![
                covenant.0.into(),
                Covenant::std_ed25519_pk_legacy(public_key).0.into(),
            ],
            sigs: vec![],
        };

        let second_transaction_result: Result<(), StateError> = state.apply_tx(&second_transaction);

        assert!(matches!(second_transaction_result, Err(ViolatesScript(_))));
    }

    // This requires creating minting transactions, which will be covered later during melswap testing.
    // Remaining variants of the `state::StateError` enum are:
    // `InvalidMelPoW`, `WrongHeader`, `CoinLocked`, and `DuplicateTx`
    // #[test]
    // fn invalid_mel_proof_of_work() {
    //     let mut state: State<InMemoryCas> = create_state(&HashMap::new(), 0);
    //
    //     let covenant: Covenant = Covenant::from_ops(&[
    //         OpCode::PushI(1_u8.into()),
    //         OpCode::PushI(2_u8.into()),
    //         OpCode::Add,
    //         OpCode::PushI(3_u8.into()),
    //         OpCode::Eql,
    //     ])
    //         .expect("Failed to create a Add covenant.");
    //
    //     let covenant_hash: themelio_structs::Address = covenant.hash();
    //
    //     let first_transaction: Transaction = Transaction {
    //         kind: TxKind::Faucet,
    //         inputs: vec![],
    //         outputs: vec![
    //             CoinData {
    //                 covhash: covenant_hash,
    //                 value: 1000.into(),
    //                 denom: Denom::Mel,
    //                 additional_data: vec![],
    //             },
    //         ],
    //         data: vec![],
    //         fee: CoinValue(20000),
    //         covenants: vec![],
    //         sigs: vec![],
    //     };
    //
    //     let _first_transaction_result: () = state.apply_tx(&first_transaction).unwrap();
    //
    //
    //     let second_transaction: Transaction = Transaction {
    //         kind: TxKind::Normal,
    //         inputs: vec![first_transaction.output_coinid(0)],
    //         outputs: vec![],
    //         data: vec![],
    //         fee: CoinValue(1000),
    //         covenants: vec![covenant.0],
    //         sigs: vec![],
    //     };
    //
    //     let second_transaction_result: Result<(), StateError> = state.apply_tx(&second_transaction);
    //
    //     assert_eq!(second_transaction_result, Err(InvalidMelPoW));
    // }

    #[test]
    #[should_panic]
    fn overflow_coins() {
        let mut state: State<InMemoryCas> = create_state(&HashMap::new(), 0);

        // We attempt to cause an overflow by creating a ton of coins, all with the maximum allowed value, then trying to spend them in one transaction.
        let faucet_coin_ids: Vec<CoinID> = (0..300)
            .into_iter()
            .map(|_index| {
                // Every iteration, we make a faucet that creates a max-value coin that any transaction can spend.
                let faucet_transaction: Transaction = Transaction {
                    kind: TxKind::Faucet,
                    inputs: vec![],
                    outputs: vec![CoinData {
                        value: MAX_COINVAL,
                        denom: Denom::Mel,
                        covhash: Covenant::always_true().hash(),
                        additional_data: vec![].into(),
                    }],
                    // random data so that the transactions are distinct (this avoids a DuplicateTx error)
                    data: vec![0; 32]
                        .tap_mut(|v| rand::thread_rng().fill_bytes(v))
                        .into(),
                    fee: CoinValue(100000),
                    covenants: vec![],
                    sigs: vec![],
                };

                state.apply_tx(&faucet_transaction).unwrap();

                faucet_transaction.output_coinid(0)
            })
            .collect();

        // Try to spend them all.
        let second_transaction: Transaction = Transaction {
            kind: TxKind::Normal,
            inputs: faucet_coin_ids,
            outputs: vec![CoinData {
                value: CoinValue(12345), // output is incorrect, but this is intentional. there are too many inputs to fit in a 128. the intention is that in the process of trying to balance the input and output, we want to trigger an overflow panic instead of correctly concluding that the sides do not balance and returning an unbalanced in out error.
                denom: Denom::Mel,
                additional_data: vec![].into(),
                covhash: Covenant::always_true().hash(),
            }],
            data: vec![].into(),
            fee: CoinValue(0), // Because we are spending so many more coins than we are creating, our transaction is free (since it reduces long-term storage burden to the network).
            covenants: vec![Covenant::always_true().0.into()],
            sigs: vec![],
        };

        // Print out the error. There needs to be an error!
        dbg!(state.apply_tx(&second_transaction).unwrap_err());
    }

    #[test]
    fn nonexistent_script() {
        let mut state: State<InMemoryCas> = create_state(&HashMap::new(), 0);

        let (public_key, _secret_key): (tmelcrypt::Ed25519PK, tmelcrypt::Ed25519SK) =
            tmelcrypt::ed25519_keygen();

        let covenant_hash: themelio_structs::Address =
            Covenant::std_ed25519_pk_legacy(public_key).hash();

        let first_transaction: Transaction = Transaction {
            kind: TxKind::Faucet,
            inputs: vec![],
            outputs: vec![CoinData {
                covhash: covenant_hash,
                value: 20000.into(),
                denom: Denom::Mel,
                additional_data: vec![].into(),
            }],
            data: vec![].into(),
            fee: CoinValue(20000),
            covenants: vec![],
            sigs: vec![],
        };

        state.apply_tx(&first_transaction).unwrap();

        let mut covenant: Covenant = Covenant::from_ops(&[
            OpCode::PushI(1_u8.into()),
            OpCode::PushI(2_u8.into()),
            OpCode::Add,
            OpCode::PushI(3_u8.into()),
            OpCode::Eql,
        ])
        .expect("Failed to create an Add covenant.");

        covenant.0 = vec![1];

        let second_transaction: Transaction = Transaction {
            kind: TxKind::Normal,
            inputs: vec![first_transaction.output_coinid(0)],
            outputs: vec![],
            data: vec![].into(),
            fee: CoinValue(1000),
            covenants: vec![covenant.0.into()],
            sigs: vec![],
        };

        let second_transaction_result: Result<(), StateError> = state.apply_tx(&second_transaction);

        assert!(matches!(
            second_transaction_result,
            Err(NonexistentScript(_))
        ));
    }

    #[test]
    fn nonexistant_coin() {
        let state: State<InMemoryCas> = create_state(&HashMap::new(), 0);

        let first_transaction: Transaction = Transaction {
            kind: TxKind::Faucet,
            inputs: vec![],
            outputs: vec![],
            data: vec![].into(),
            fee: CoinValue(1000),
            covenants: vec![],
            sigs: vec![],
        };

        state.clone().apply_tx(&first_transaction).unwrap();

        let second_transaction: Transaction = Transaction {
            kind: TxKind::Normal,
            inputs: vec![first_transaction.output_coinid(0)],
            outputs: vec![],
            data: vec![].into(),
            fee: CoinValue(1000),
            covenants: vec![],
            sigs: vec![],
        };

        let second_transaction_result: Result<(), StateError> =
            state.clone().apply_tx(&second_transaction);

        assert!(matches!(second_transaction_result, Err(NonexistentCoin(_))));
    }

    #[test]
    fn insufficient_fees() {
        let state: State<InMemoryCas> = create_state(&HashMap::new(), 0);

        let covenant: Covenant = Covenant::from_ops(&[
            OpCode::PushI(1_u8.into()),
            OpCode::PushI(2_u8.into()),
            OpCode::Add,
            OpCode::PushI(3_u8.into()),
            OpCode::Eql,
        ])
        .expect("Failed to create a Add covenant.");

        let transaction: Transaction = Transaction {
            kind: TxKind::Faucet,
            inputs: vec![],
            outputs: vec![],
            data: vec![].into(),
            fee: CoinValue(1000),
            covenants: vec![covenant.0.into()],
            sigs: vec![],
        };

        let transaction_result: Result<(), StateError> =
            state.clone().apply_tx_batch(&[transaction]);

        assert_eq!(transaction_result, Err(InsufficientFees(CoinValue(1861))));
    }

    #[test]
    fn unbalanced_input_and_output() {
        let state: State<InMemoryCas> = create_state(&HashMap::new(), 0);

        let spend_nonexistant_coins_transaction: Transaction = Transaction {
            kind: TxKind::Normal,
            inputs: vec![],
            outputs: vec![],
            data: vec![].into(),
            fee: CoinValue(1000),
            covenants: vec![],
            sigs: vec![],
        };

        let spend_nonexistant_coins_transaction_result: Result<(), StateError> = state
            .clone()
            .apply_tx_batch(&[spend_nonexistant_coins_transaction]);

        assert_eq!(
            spend_nonexistant_coins_transaction_result,
            Err(UnbalancedInOut)
        );
    }

    #[test]
    fn apply_batch_normal() {
        let state = create_state(&HashMap::new(), 0);
        let txx = valid_txx(tmelcrypt::ed25519_keygen());
        // all at once
        state.clone().apply_tx_batch(&txx).unwrap();
        // not all at once
        {
            let mut state = state.clone();
            for tx in txx.iter() {
                state.apply_tx(tx).unwrap();
            }
        }
        // now shuffle
        let mut txx = txx;
        txx.shuffle(&mut rand::thread_rng());
        // all at once must also work
        state.clone().apply_tx_batch(&txx).unwrap();
        let mut state = state;
        for tx in txx.iter() {
            if state.apply_tx(tx).is_err() {
                return;
            }
        }
        panic!("should not reach here")
    }

    #[test]
    fn fee_pool_increase() {
        let mut state = create_state(&HashMap::new(), 0);
        let txx = valid_txx(tmelcrypt::ed25519_keygen());
        for tx in txx.iter() {
            let pre_fee = state.fee_pool + state.tips;
            state.apply_tx(tx).unwrap();
            assert_eq!(pre_fee + tx.fee, state.fee_pool + state.tips);
        }
    }

    #[test]
    fn forbid_mainnet_faucet() {
        let mut state = create_state(&HashMap::new(), 0);
        state.fee_multiplier = 0;
        state.network = NetID::Mainnet;
        state
            .apply_tx(&Transaction {
                kind: TxKind::Faucet,
                inputs: vec![],
                outputs: vec![],
                data: vec![].into(),
                fee: CoinValue(1000),
                covenants: vec![],
                sigs: vec![],
            })
            .unwrap_err();
    }

    #[test]
    fn no_duplicate_faucet_same_block() {
        let mut state = create_state(&HashMap::new(), 0);
        state.network = NetID::Testnet;
        let faucet = TransactionBuilder::new()
            .kind(TxKind::Faucet)
            .output(CoinData {
                denom: Denom::Mel,
                covhash: Address(Default::default()),
                value: CoinValue(100000),
                additional_data: vec![].into(),
            })
            .fee(CoinValue(20000))
            .build()
            .unwrap();
        state.apply_tx(&faucet).unwrap();

        assert_eq!(state.apply_tx(&faucet), Err(StateError::DuplicateTx))
    }

    #[test]
    fn no_duplicate_faucet_diff_blocks() {
        let mut state = create_state(&HashMap::new(), 0);
        state.network = NetID::Testnet;
        let faucet = TransactionBuilder::new()
            .kind(TxKind::Faucet)
            .output(CoinData {
                denom: Denom::Mel,
                covhash: Address(Default::default()),
                value: CoinValue(100000),
                additional_data: vec![].into(),
            })
            .fee(CoinValue(20000))
            .build()
            .unwrap();
        state.apply_tx(&faucet).unwrap();
        state = state.seal(None).next_state();
        assert_eq!(state.apply_tx(&faucet), Err(StateError::DuplicateTx))
    }

    #[test]
    fn staked_coin_cannot_spend() {
        let mut state = create_state(&HashMap::new(), 0);
        state.fee_multiplier = 0;
        state = state.seal(None).next_state();
        // create some syms
        let sym_faucet = Transaction {
            kind: TxKind::Faucet,
            inputs: vec![],
            outputs: vec![
                CoinData {
                    denom: Denom::Sym,
                    value: CoinValue(1000),
                    covhash: Covenant::always_true().hash(),
                    additional_data: vec![].into(),
                },
                CoinData {
                    denom: Denom::Mel,
                    value: CoinValue(1000),
                    covhash: Covenant::always_true().hash(),
                    additional_data: vec![].into(),
                },
            ],
            fee: CoinValue(0),
            covenants: vec![],
            data: vec![].into(),
            sigs: vec![],
        };
        state.apply_tx(&sym_faucet).unwrap();
        // stake the syms
        let sym_stake = TransactionBuilder::new()
            .kind(TxKind::Stake)
            .input(sym_faucet.output_coinid(0), sym_faucet.outputs[0].clone())
            .input(sym_faucet.output_coinid(1), sym_faucet.outputs[1].clone())
            .output(sym_faucet.outputs[0].clone())
            .covenant(Covenant::always_true().0.into())
            .output(sym_faucet.outputs[1].clone())
            .data(
                StakeDoc {
                    pubkey: tmelcrypt::Ed25519PK(Default::default()),
                    e_start: 1,
                    e_post_end: 2,
                    syms_staked: CoinValue(1000),
                }
                .stdcode()
                .into(),
            )
            .build()
            .unwrap();
        state.apply_tx(&sym_stake).unwrap();
        let another = TransactionBuilder::new()
            .input(sym_stake.output_coinid(0), sym_stake.outputs[0].clone())
            .input(sym_stake.output_coinid(1), sym_stake.outputs[1].clone())
            .output(CoinData {
                denom: Denom::Sym,
                value: CoinValue(1000),
                covhash: Covenant::always_true().hash(),
                additional_data: vec![].into(),
            })
            .covenant(Covenant::always_true().0.into())
            .fee(CoinValue(1000))
            .build()
            .unwrap();
        assert_eq!(
            StateError::CoinLocked,
            state.apply_tx(&another).unwrap_err()
        );
    }

    #[test]
    fn simple_dmt() {
        // custom08 has all the tips
        let mut test_state = create_state(&HashMap::new(), 0);
        test_state.network = NetID::Custom08;
        // insert a bunch of transactions, then make sure all of them have valid proofs of inclusion
        let txx_to_insert = valid_txx(tmelcrypt::ed25519_keygen());
        for tx in txx_to_insert.iter() {
            test_state.apply_tx(tx).unwrap();
        }
        let sealed = test_state.seal(None);
        let header = sealed.header();
        let dmt = sealed.0.tip908_transactions();
        for tx in txx_to_insert.iter() {
            let posn = sealed.0.transaction_sorted_posn(tx.hash_nosigs()).unwrap();
            let proof = dmt.proof(posn);
            assert!(novasmt::dense::verify_dense(
                &proof,
                header.transactions_hash.0,
                posn,
                novasmt::hash_data(
                    &tx.hash_nosigs()
                        .0
                        .to_vec()
                        .tap_mut(|v| v.extend_from_slice(&tx.stdcode().hash().0))
                ),
            ));
        }
    }
}
