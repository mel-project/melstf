mod applytx;
mod coins;
pub(crate) mod melmint;

pub use crate::stake::*;
use crate::tip_heights::TIP_902_HEIGHT;
use crate::{
    smtmapping::*,
    state::applytx::apply_tx_batch_impl,
    tip_heights::{
        TIP_901_HEIGHT, TIP_906_HEIGHT, TIP_908_HEIGHT, TIP_909A_HEIGHT, TIP_909_HEIGHT,
    },
};

use std::collections::BTreeMap;
use std::fmt::Debug;

use crate::state::melmint::PoolMapping;
use derivative::Derivative;
use novasmt::{dense::DenseMerkleTree, ContentAddrStore, Database, InMemoryCas};
use stdcode::StdcodeSerializeExt;
use tap::Pipe;
use themelio_structs::{
    Address, Block, BlockHeight, CoinData, CoinDataHeight, CoinID, CoinValue, ConsensusProof,
    Denom, Header, NetID, PoolKey, ProposerAction, Transaction, TxHash, STAKE_EPOCH,
};
use thiserror::Error;
use tmelcrypt::{HashVal, Hashable};

pub use self::coins::CoinMapping;

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
    pub fn tip_901(&self) -> bool {
        self.tip_condition(TIP_901_HEIGHT)
    }

    /// Returns true iff TIP 902 rule changes apply.
    pub fn tip_902(&self) -> bool {
        self.tip_condition(TIP_902_HEIGHT)
    }

    /// Returns true iff TIP 906 rule changes apply.
    pub fn tip_906(&self) -> bool {
        self.tip_condition(TIP_906_HEIGHT)
    }

    /// Returns true iff TIP 908 rule changes apply.
    pub fn tip_908(&self) -> bool {
        self.tip_condition(TIP_908_HEIGHT) || self.network == NetID::Custom08
    }

    /// Returns true iff TIP 909 rule changes apply.
    pub fn tip_909(&self) -> bool {
        self.tip_condition(TIP_909_HEIGHT)
    }

    /// Returns true iff TIP 909a rule changes apply.
    pub fn tip_909a(&self) -> bool {
        self.tip_condition(TIP_909A_HEIGHT)
    }

    /// Applies a single transaction.
    pub fn apply_tx(&mut self, tx: &Transaction) -> Result<(), StateError> {
        self.apply_tx_batch(std::slice::from_ref(tx))
    }

    /// Applies a whole lot of transactions.
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

    /// Calculates the "transactions" root hash. Note that this is different depending on whether the block is pre-tip908 or post-tip908.
    pub fn transactions_root_hash(&self) -> HashVal {
        if self.tip_908() {
            HashVal(self.tip908_transactions().root_hash())
        } else {
            let db = Database::new(InMemoryCas::default());
            let mut smt = SmtMapping::new(db.get_tree(Default::default()).unwrap());
            for (k, v) in self.transactions.iter() {
                smt.insert(*k, v.clone());
            }
            smt.root_hash()
        }
    }

    /// Obtains the dense merkle tree (tip-908)
    pub fn tip908_transactions(&self) -> DenseMerkleTree {
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
        DenseMerkleTree::new(&vv)
    }

    /// Obtains the sorted position of the given transaction within this state.
    pub fn transaction_sorted_posn(&self, txhash: TxHash) -> Option<usize> {
        self.transactions
            .keys()
            .enumerate()
            .find(|(_, k)| k == &&txhash)
            .map(|(i, _)| i)
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
                log::warn!("{:?}", tx.kind);
            });

            log::warn!("pre-apply header: {:#?}", self.header());
            log::warn!("block: {:#?}", block);

            Err(StateError::WrongHeader)
        } else {
            Ok(basis)
        }
    }

    /// Confirms a state with a given consensus proof. If called with a second argument, this function is supposed to be called to *verify* the consensus proof.
    ///
    /// **TODO**: Right now it DOES NOT check the consensus proof!
    #[deprecated]
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
    use std::collections::HashMap;

    use rand::prelude::SliceRandom;
    use stdcode::StdcodeSerializeExt;
    use tap::Tap;
    use themelio_structs::{
        Address, BlockHeight, CoinData, CoinID, CoinValue, Denom, NetID, StakeDoc, Transaction,
        TransactionBuilder, TxKind, MAX_COINVAL,
    };
    use tmelcrypt::Hashable;

    use crate::{
        melvm::opcode::OpCode,
        melvm::Covenant,
        testing::functions::{create_state, valid_txx},
        State, StateError,
        StateError::{
            InsufficientFees, InvalidMelPoW, MalformedTx, NonexistentCoin, NonexistentScript,
            UnbalancedInOut, ViolatesScript,
        tip_heights::TIP_908_HEIGHT,
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
                additional_data: vec![],
            }],
            data: vec![],
            fee: CoinValue(20000),
            covenants: vec![],
            sigs: vec![],
        };

        let _first_transaction_result: () = state.apply_tx(&first_transaction).unwrap();

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
            data: vec![],
            fee: CoinValue(1000),
            covenants: vec![covenant.0, Covenant::std_ed25519_pk_legacy(public_key).0],
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
        let faucet_coin_ids: Vec<CoinID> = (0..300).into_iter().map(|index| {
            // Every iteration, we make a faucet that creates a max-value coin that any transaction can spend.
            let faucet_transaction: Transaction = Transaction {
                kind: TxKind::Faucet,
                inputs: vec![],
                outputs: vec![CoinData {
                    value: MAX_COINVAL,
                    denom: Denom::Mel,
                    covhash: Covenant::always_true().hash(),
                    additional_data: vec![],
                }],
                // random data so that the transactions are distinct (this avoids a DuplicateTx error)
                data: vec![0; 32].tap_mut(|v| rand::thread_rng().fill_bytes(v)),
                fee: CoinValue(100000),
                covenants: vec![],
                sigs: vec![],
            };

            let _first_transaction_result: () = state.apply_tx(&faucet_transaction).unwrap();

            faucet_transaction.output_coinid(0)
        }).collect();


        // Try to spend them all.
        let second_transaction: Transaction = Transaction {
            kind: TxKind::Normal,
            inputs: faucet_coin_ids,
            outputs: vec![CoinData {
                value: CoinValue(12345), // output is incorrect, but this is intentional. there are too many inputs to fit in a 128. the intention is that in the process of trying to balance the input and output, we want to trigger an overflow panic instead of correctly concluding that the sides do not balance and returning an unbalanced in out error.
                denom: Denom::Mel,
                additional_data: vec![],
                covhash: Covenant::always_true().hash(),
            }],
            data: vec![],
            fee: CoinValue(0), // Because we are spending so many more coins than we are creating, our transaction is free (since it reduces long-term storage burden to the network).
            covenants: vec![Covenant::always_true().0],
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
                additional_data: vec![],
            }],
            data: vec![],
            fee: CoinValue(20000),
            covenants: vec![],
            sigs: vec![],
        };

        let _first_transaction_result: () = state.apply_tx(&first_transaction).unwrap();

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
            data: vec![],
            fee: CoinValue(1000),
            covenants: vec![covenant.0],
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
            data: vec![],
            fee: CoinValue(1000),
            covenants: vec![],
            sigs: vec![],
        };

        let _first_transaction_result: () = state.clone().apply_tx(&first_transaction).unwrap();

        let second_transaction: Transaction = Transaction {
            kind: TxKind::Normal,
            inputs: vec![first_transaction.output_coinid(0)],
            outputs: vec![],
            data: vec![],
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
            data: vec![],
            fee: CoinValue(1000),
            covenants: vec![covenant.0],
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
            data: vec![],
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
                data: vec![],
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
                additional_data: vec![],
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
                additional_data: vec![],
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
                    additional_data: vec![],
                },
                CoinData {
                    denom: Denom::Mel,
                    value: CoinValue(1000),
                    covhash: Covenant::always_true().hash(),
                    additional_data: vec![],
                },
            ],
            fee: CoinValue(0),
            covenants: vec![],
            data: vec![],
            sigs: vec![],
        };
        state.apply_tx(&sym_faucet).unwrap();
        // stake the syms
        let sym_stake = TransactionBuilder::new()
            .kind(TxKind::Stake)
            .input(sym_faucet.output_coinid(0), sym_faucet.outputs[0].clone())
            .input(sym_faucet.output_coinid(1), sym_faucet.outputs[1].clone())
            .output(sym_faucet.outputs[0].clone())
            .covenant(Covenant::always_true().0)
            .output(sym_faucet.outputs[1].clone())
            .data(
                StakeDoc {
                    pubkey: tmelcrypt::Ed25519PK(Default::default()),
                    e_start: 1,
                    e_post_end: 2,
                    syms_staked: CoinValue(1000),
                }
                .stdcode(),
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
                additional_data: vec![],
            })
            .covenant(Covenant::always_true().0)
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
        let dmt = sealed.inner_ref().tip908_transactions();
        for tx in txx_to_insert.iter() {
            let posn = sealed
                .inner_ref()
                .transaction_sorted_posn(tx.hash_nosigs())
                .unwrap();
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
