use crate::{
    melpow,
    melvm::{Address, CovenantEnv},
    stake::StakeDoc,
    state::melmint,
    BlockHeight, CoinData, CoinDataHeight, CoinID, CoinValue, Denom, NetID, State, StateError,
    Transaction, TxHash, TxKind,
};

use std::convert::TryInto;

use dashmap::DashMap;
use parking_lot::Mutex;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tmelcrypt::HashVal;

/// A mutable "handle" to a particular State. Can be "committed" like a database transaction.
pub(crate) struct StateHandle<'a> {
    state: &'a mut State,

    coin_cache: DashMap<CoinID, Option<CoinDataHeight>>,
    transactions_cache: DashMap<TxHash, Transaction>,

    fee_pool_cache: CoinValue,
    tips_cache: CoinValue,

    dosc_speed_cache: Mutex<u128>,

    stakes_cache: DashMap<TxHash, StakeDoc>,
}

fn faucet_dedup_pseudocoin(txhash: TxHash) -> CoinID {
    CoinID {
        txhash: tmelcrypt::hash_keyed(b"fdp", &txhash.0).into(),
        index: 0,
    }
}

impl<'a> StateHandle<'a> {
    /// Creates a new state handle.
    pub fn new(state: &'a mut State) -> Self {
        let fee_pool_cache = state.fee_pool;
        let tips_cache = state.tips;
        let dosc_speed = state.dosc_speed;
        StateHandle {
            state,

            coin_cache: DashMap::new(),
            transactions_cache: DashMap::new(),

            fee_pool_cache,
            tips_cache,

            dosc_speed_cache: Mutex::new(dosc_speed),

            stakes_cache: DashMap::new(),
        }
    }

    /// Applies a batch of transactions, returning an error if any of them fail. Consumes and re-returns the handle; if any fail the handle is gone.
    pub fn apply_tx_batch(mut self, txx: &[Transaction]) -> Result<Self, StateError> {
        let was_batch_successful: Result<(), crate::state::StateError> =
            txx.iter().try_for_each(|tx| {
                let is_transaction_a_faucet: bool = tx.kind == TxKind::Faucet;

                if is_transaction_a_faucet {
                    let pseudocoin = faucet_dedup_pseudocoin(tx.hash_nosigs());

                    if self.state.coins.get(&pseudocoin).0.is_some() {
                        Err(StateError::DuplicateTx)
                    } else {
                        self.state.coins.insert(
                            pseudocoin,
                            CoinDataHeight {
                                coin_data: CoinData {
                                    denom: Denom::Mel,
                                    value: 0.into(),
                                    additional_data: vec![],
                                    covhash: HashVal::default().into(),
                                },
                                height: 0.into(),
                            },
                        );

                        Ok(())
                    }
                } else if !tx.is_well_formed() {
                    Err(StateError::MalformedTx)
                } else if is_transaction_a_faucet && self.state.network == NetID::Mainnet {
                    Err(StateError::UnbalancedInOut)
                } else {
                    self.transactions_cache.insert(tx.hash_nosigs(), tx.clone());

                    self.apply_tx_fees(tx)
                }
            });

        match was_batch_successful {
            Ok(()) => {
                // apply specials in parallel
                txx.par_iter()
                    .filter(|tx| tx.kind != TxKind::Normal && tx.kind != TxKind::Faucet)
                    .map(|tx| self.apply_tx_special(tx))
                    .collect::<Result<_, _>>()?;
                // apply outputs in parallel
                txx.par_iter().for_each(|tx| self.apply_tx_outputs(tx));
                // apply inputs in parallel
                txx.par_iter()
                    .map(|tx| self.apply_tx_inputs(tx))
                    .collect::<Result<_, _>>()?;

                Ok(self)
            }
            Err(error) => Err(error),
        }
    }

    /// Commits all the changes in this handle, at once.
    pub fn commit(self) {
        // commit coins
        self.coin_cache.into_iter().for_each(|(key, value)| {
            if let Some(value) = value {
                self.state.coins.insert(key, value);
            } else {
                self.state.coins.delete(&key);
            }
        });

        // commit txx
        self.transactions_cache
            .into_iter()
            .for_each(|(key, value)| {
                self.state.transactions.insert(key, value);
            });

        // commit fees
        self.state.fee_pool = self.fee_pool_cache;
        self.state.tips = self.tips_cache;

        // commit stakes
        self.stakes_cache.into_iter().for_each(|(key, value)| {
            self.state.stakes.insert(key, value);
        });

        self.state.dosc_speed = *self.dosc_speed_cache.lock()
    }

    fn apply_tx_inputs(&self, tx: &Transaction) -> Result<(), StateError> {
        // let mut output: Vec<u8> = Vec::new();
        // // go through output
        // let was_encoding_successful: Result<(), EncodeError> = ops.iter().try_for_each(|op| {
        //     match op.encode() {
        //         Ok(data) => {
        //             Ok(output.extend_from_slice(&data))
        //         },
        //         Err(error) => Err(error),
        //     }
        // });
        //
        // match was_encoding_successful {
        //     Ok(()) => Ok(Covenant(output)),
        //     Err(error) => Err(error),
        // }

        let scripts = tx.script_as_map();
        // build a map of input coins
        let mut in_coins: imbl::HashMap<Denom, CoinValue> = imbl::HashMap::new();
        // get last header
        let last_header = self
            .state
            .history
            .get(&BlockHeight(self.state.height.0.saturating_sub(1)))
            .0
            .unwrap_or_else(|| self.state.clone().seal(None).header());

        // iterate through the inputs
        let were_inputs_successful: Result<(), StateError> = tx
            .inputs
            .iter()
            .enumerate()
            .try_for_each(|(spend_idx, coin_id)| {
                if self.get_stake(coin_id.txhash).is_some() && coin_id.index == 0 {
                    // only the first output is locked
                    Err(StateError::CoinLocked)
                } else {
                    let coin_data = self.get_coin(*coin_id);

                    match coin_data {
                        None => Err(StateError::NonexistentCoin(*coin_id)),
                        Some(coin_data) => {
                            log::trace!(
                                "coin_data {:?} => {:?} for txid {:?}",
                                coin_id,
                                coin_data,
                                tx.hash_nosigs()
                            );
                            let script = scripts.get(&coin_data.coin_data.covhash).ok_or(
                                StateError::NonexistentScript(coin_data.coin_data.covhash),
                            )?;

                            let was_check_an_error: bool = !script.check(
                                tx,
                                CovenantEnv {
                                    parent_coinid: coin_id,
                                    parent_cdh: &coin_data,
                                    spender_index: spend_idx as u8,
                                    last_header: &last_header,
                                },
                            );

                            if was_check_an_error {
                                Err(StateError::ViolatesScript(coin_data.coin_data.covhash))
                            } else {
                                self.del_coin(*coin_id);

                                in_coins.insert(
                                    coin_data.coin_data.denom,
                                    in_coins
                                        .get(&coin_data.coin_data.denom)
                                        .cloned()
                                        .unwrap_or_else(|| 0u128.into())
                                        + coin_data.coin_data.value,
                                );

                                Ok(())
                            }
                        }
                    }
                }
            });

        match were_inputs_successful {
            Err(error) => Err(error),
            Ok(()) => {
                // balance inputs and outputs. ignore outputs with empty cointype (they create a new token kind)
                let out_coins = tx.total_outputs();

                let is_transaction_not_a_faucet: bool = tx.kind != TxKind::Faucet;

                if is_transaction_not_a_faucet {
                    out_coins.iter().try_for_each(|(currency, value)| {
                        // we skip the created doscs for a DoscMint transaction
                        let is_transaction_a_dosc: bool =
                            tx.kind == TxKind::DoscMint && *currency == Denom::NomDosc;

                        if !is_transaction_a_dosc {
                            let in_value = in_coins
                                .get(currency)
                                .copied()
                                .unwrap_or_else(|| u128::MAX.into());

                            let unbalanced_check: bool =
                                *currency != Denom::NewCoin && *value != in_value;

                            if unbalanced_check {
                                log::warn!(
                                    "unbalanced: {} {:?} in, {} {:?} out",
                                    in_value,
                                    currency,
                                    value,
                                    currency
                                );

                                Err(StateError::UnbalancedInOut)
                            } else {
                                Ok(())
                            }
                        } else {
                            Ok(())
                        }
                    })
                } else {
                    Ok(())
                }
            }
        }
    }

    fn apply_tx_fees(&mut self, tx: &Transaction) -> Result<(), StateError> {
        // fees
        let min_fee = tx.base_fee(self.state.fee_multiplier, 0);
        if tx.fee < min_fee {
            Err(StateError::InsufficientFees(min_fee))
        } else {
            let tips = tx.fee - min_fee;
            self.tips_cache.0 = self.tips_cache.0.saturating_add(tips.0);
            self.fee_pool_cache.0 = self.fee_pool_cache.0.saturating_add(min_fee.0);

            Ok(())
        }
    }

    fn apply_tx_outputs(&self, tx: &Transaction) {
        let height = self.state.height;

        tx.outputs
            .iter()
            .enumerate()
            .for_each(|(index, coin_data)| {
                let mut coin_data = coin_data.clone();
                if coin_data.denom == Denom::NewCoin {
                    coin_data.denom = Denom::Custom(tx.hash_nosigs());
                }
                // if covenant hash is zero, this destroys the coins permanently
                if coin_data.covhash != Address::coin_destroy() {
                    self.set_coin(
                        CoinID {
                            txhash: tx.hash_nosigs(),
                            index: index.try_into().unwrap(),
                        },
                        CoinDataHeight { coin_data, height },
                    );
                }
            });
    }

    fn apply_tx_special(&self, tx: &Transaction) -> Result<(), StateError> {
        match tx.kind {
            TxKind::DoscMint => self.apply_tx_special_doscmint(tx),
            TxKind::Stake => self.apply_tx_special_stake(tx),
            _ => Ok(()),
        }
    }

    fn apply_tx_special_doscmint(&self, tx: &Transaction) -> Result<(), StateError> {
        let coin_id = *tx.inputs.get(0).ok_or(StateError::MalformedTx).unwrap();
        let coin_data = self.get_coin(coin_id).ok_or(StateError::MalformedTx)?;
        // make sure the time is long enough that we can easily measure it
        if (self.state.height - coin_data.height).0 < 100 {
            log::warn!("too recent");
            return Err(StateError::InvalidMelPoW);
        }
        // construct puzzle seed
        let chi = tmelcrypt::hash_keyed(
            &self.state.history.get(&coin_data.height).0.unwrap().hash(),
            &stdcode::serialize(tx.inputs.get(0).ok_or(StateError::MalformedTx).unwrap()).unwrap(),
        );
        // get difficulty and proof
        let (difficulty, proof_bytes): (u32, Vec<u8>) =
            stdcode::deserialize(&tx.data).map_err(|_| StateError::MalformedTx)?;
        let proof = melpow::Proof::from_bytes(&proof_bytes)
            .ok_or(StateError::MalformedTx)
            .unwrap();
        if !proof.verify(&chi, difficulty as _) {
            log::warn!("chi = {}", chi);
            log::warn!(
                "proof = {} ({:?})",
                tmelcrypt::hash_single(&proof_bytes),
                difficulty
            );
            log::warn!("proof verification failed");
            return Err(StateError::InvalidMelPoW);
        }
        // compute speeds
        let my_speed = 2u128.pow(difficulty) / (self.state.height - coin_data.height).0 as u128;
        let reward_real = melmint::calculate_reward(
            my_speed,
            self.state
                .history
                .get(&BlockHeight(self.state.height.0 - 1))
                .0
                .unwrap()
                .dosc_speed,
            difficulty,
        );
        {
            let mut dosc_speed = self.dosc_speed_cache.lock();
            *dosc_speed = dosc_speed.max(my_speed);
        }
        let reward_nom = CoinValue(melmint::dosc_inflate_r2n(self.state.height, reward_real));
        // ensure that the total output of DOSCs is correct
        let total_dosc_output = tx
            .total_outputs()
            .get(&Denom::NomDosc)
            .cloned()
            .unwrap_or_default();
        if total_dosc_output > reward_nom {
            return Err(StateError::InvalidMelPoW);
        }
        Ok(())
    }
    fn apply_tx_special_stake(&self, tx: &Transaction) -> Result<(), StateError> {
        // first we check that the data is correct
        let stake_doc: StakeDoc =
            stdcode::deserialize(&tx.data).map_err(|_| StateError::MalformedTx)?;
        let curr_epoch = self.state.height.epoch();
        // then we check that the first coin is valid
        let first_coin = tx.outputs.get(0).ok_or(StateError::MalformedTx)?;

        let is_first_coin_not_a_sym: bool = first_coin.denom != Denom::Sym;

        if is_first_coin_not_a_sym {
            Err(StateError::MalformedTx)
        // then we check consistency
        } else if stake_doc.e_start > curr_epoch
            && stake_doc.e_post_end > stake_doc.e_start
            && stake_doc.syms_staked == first_coin.value
        {
            self.set_stake(tx.hash_nosigs(), stake_doc);
            log::warn!("**** ADDING STAKER {:?} ****", stake_doc);
            Ok(())
        } else {
            log::warn!("**** REJECTING STAKER {:?} ****", stake_doc);
            Ok(())
        }
    }

    fn get_coin(&self, coin_id: CoinID) -> Option<CoinDataHeight> {
        self.coin_cache
            .entry(coin_id)
            .or_insert_with(|| self.state.coins.get(&coin_id).0)
            .value()
            .clone()
    }

    fn set_coin(&self, coin_id: CoinID, value: CoinDataHeight) {
        self.coin_cache.insert(coin_id, Some(value));
    }

    fn del_coin(&self, coin_id: CoinID) {
        self.coin_cache.insert(coin_id, None);
    }

    fn get_stake(&self, txhash: TxHash) -> Option<StakeDoc> {
        if let Some(cached_sd) = self.stakes_cache.get(&txhash).as_deref() {
            Some(cached_sd).cloned()
        } else if let Some(sd) = self.state.stakes.get(&txhash).0 {
            self.stakes_cache.insert(txhash, sd)
        } else {
            None
        }
    }

    fn set_stake(&self, txhash: TxHash, sdoc: StakeDoc) {
        self.stakes_cache.insert(txhash, sdoc);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    // use crate::melvm::Covenant;
    // use crate::state::applytx::StateHandle;
    // // use crate::testing::factory::*;
    // use crate::testing::fixtures::*;
    // use crate::{CoinData, CoinID, State};
    // use rstest::*;
    // use tmelcrypt::{Ed25519PK, Ed25519SK};
    //
    // #[rstest]
    // fn test_apply_tx_inputs_single_valid_tx(
    //     genesis_state: State,
    //     genesis_mel_coin_id: CoinID,
    //     genesis_mel_coin_data: CoinData,
    //     genesis_covenant_keypair: (Ed25519PK, Ed25519SK),
    //     genesis_covenant: Covenant,
    //     keypair: (Ed25519PK, Ed25519SK),
    // ) {
    //     // Init state and state handle
    //     let mut state = genesis_state.clone();
    //     let state_handle = StateHandle::new(&mut state);
    //
    //     // Create a valid signed transaction from first coin
    //     // let fee = 3000000;
    //     // let tx = tx_factory(
    //     //     TxKind::Normal,
    //     //     genesis_covenant_keypair,
    //     //     keypair.0,
    //     //     genesis_mel_coin_id,
    //     //     genesis_covenant,
    //     //     genesis_mel_coin_data.value,
    //     //     fee,
    //     // );
    //     //
    //     // // Apply tx inputs and verify no error
    //     // let res = state_handle.apply_tx_inputs(&tx);
    //     //
    //     // assert!(res.is_ok());
    // }
}
