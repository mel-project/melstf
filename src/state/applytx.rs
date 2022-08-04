use std::time::Instant;

use novasmt::ContentAddrStore;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::{FxHashMap, FxHashSet};
use themelio_structs::{
    Address, BlockHeight, CoinData, CoinDataHeight, CoinID, CoinValue, Denom, NetID, StakeDoc,
    Transaction, TxHash, TxKind,
};
use tmelcrypt::HashVal;

use crate::{
    melmint,
    melvm::{Covenant, CovenantEnv},
    LegacyMelPowHash, State, StateError, Tip910MelPowHash,
};

/// Applies a batch of transactions to the state.
pub fn apply_tx_batch_impl<C: ContentAddrStore>(
    this: &State<C>,
    txx: &[Transaction],
) -> Result<State<C>, StateError> {
    // we first obtain *all* the relevant coins
    let relevant_coins = load_relevant_coins(this, txx)?;
    // apply the stake transactions
    let new_stakes = load_stake_info(this, txx)?;
    // check validity of every transaction, with respect to the relevant coins and stakes
    txx.par_iter()
        .try_for_each(|tx| check_tx_validity(this, tx, &relevant_coins, &new_stakes))?;
    // check the stake txx
    let new_max_speed = txx
        .par_iter()
        .filter(|tx| tx.kind == TxKind::DoscMint)
        .try_fold(
            || this.dosc_speed,
            |a, tx| {
                let new_speed = check_doscmint_validity(this, &relevant_coins, tx)?;
                Ok(a.max(new_speed))
            },
        )
        .try_reduce(|| this.dosc_speed, |a, b| Ok(a.max(b)))?;
    // great, now we create the new state
    let mut next_state = this.clone();
    for tx in txx {
        let txhash = tx.hash_nosigs();

        if tx.kind == TxKind::Faucet {
            let pseudocoin = faucet_dedup_pseudocoin(tx.hash_nosigs());
            if next_state.coins.get_coin(pseudocoin).is_some() {
                return Err(StateError::DuplicateTx);
            }
            next_state.coins.insert_coin(
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
                next_state.tip_906(),
            );
        }

        for (i, _) in tx.outputs.iter().enumerate() {
            let coinid = CoinID::new(txhash, i as u8);
            // this filters out coins that we get rid of (e.g. due to them going to the coin destruction cov)
            if let Some(coin_data) = relevant_coins.get(&coinid) {
                next_state
                    .coins
                    .insert_coin(coinid, coin_data.clone(), this.tip_906());
            }
        }
        for coinid in tx.inputs.iter() {
            next_state.coins.remove_coin(*coinid, this.tip_906());
        }

        // fees
        let min_fee = tx.base_fee(next_state.fee_multiplier, 0, |c| {
            Covenant(c.to_vec()).weight().unwrap_or(0)
        });
        if tx.fee < min_fee {
            return Err(StateError::InsufficientFees(min_fee));
        } else {
            let tips = tx.fee - min_fee;
            next_state.tips.0 = next_state.tips.0.saturating_add(tips.0);
            next_state.fee_pool.0 = next_state.fee_pool.0.saturating_add(min_fee.0);
        }
        next_state.transactions.insert(txhash, tx.clone());
    }
    // dosc
    next_state.dosc_speed = new_max_speed;
    // apply stakes
    for (k, v) in new_stakes {
        next_state.stakes.insert(k, v);
    }
    Ok(next_state)
}

fn load_relevant_coins<C: ContentAddrStore>(
    this: &State<C>,
    txx: &[Transaction],
) -> Result<FxHashMap<CoinID, CoinDataHeight>, StateError> {
    let height = this.height;

    let mut accum: FxHashMap<CoinID, CoinDataHeight> = FxHashMap::default();

    // add the ones created in this batch
    for tx in txx {
        if !tx.is_well_formed() {
            return Err(StateError::MalformedTx);
        }

        // dedup faucet
        if tx.kind == TxKind::Faucet {
            // exception to be bug-compatible with the one guy who exploited the inflation bug
            if this.network == NetID::Mainnet
                && tx.hash_nosigs().to_string()
                    != "30a60b20830f000f755b70c57c998553a303cc11f8b1f574d5e9f7e26b645d8b"
            {
                return Err(StateError::MalformedTx);
            }

            let pseudocoin = faucet_dedup_pseudocoin(tx.hash_nosigs());

            if this.coins.get_coin(pseudocoin).is_some() || accum.get(&pseudocoin).is_some() {
                return Err(StateError::DuplicateTx);
            }

            accum.insert(
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
        }

        let txhash = tx.hash_nosigs();

        let coins_to_add: FxHashMap<CoinID, CoinDataHeight> = tx.outputs.par_iter().enumerate().filter_map(|(i, coin_data)| {
            let mut coin_data = coin_data.clone();
            if coin_data.denom == Denom::NewCoin {
                coin_data.denom = Denom::Custom(tx.hash_nosigs());
            }

            // if covenant hash is zero, this destroys the coins permanently
            if coin_data.covhash != Address::coin_destroy() {
                Some((CoinID::new(txhash, i as u8), CoinDataHeight { coin_data, height }))
            } else {
                None
            }
        }).collect::<FxHashMap<CoinID, CoinDataHeight>>();

        if !coins_to_add.is_empty() {
            accum.extend(coins_to_add);
        }

    }
    // add the ones *referenced* in this batch
    let cache: FxHashMap<CoinID, Option<CoinDataHeight>> = txx
        .into_par_iter()
        .flat_map(|transaction| transaction.inputs.par_iter())
        .map(|input| (*input, this.coins.get_coin(*input)))
        .collect();

    for tx in txx {
        for input in tx.inputs.iter() {
            if !accum.contains_key(input) {
                let from_disk = cache
                    .get(input)
                    .unwrap()
                    .clone()
                    .ok_or(StateError::NonexistentCoin(*input))?;
                accum.insert(*input, from_disk);
            }
        }
    }

    // ensure no double-spending within this batch
    let mut seen = FxHashSet::default();

    for tx in txx {
        for input in tx.inputs.iter() {
            if !seen.insert(input) {
                return Err(StateError::NonexistentCoin(*input));
            }
        }
    }

    Ok(accum)
}

fn load_stake_info<C: ContentAddrStore>(
    this: &State<C>,
    txx: &[Transaction],
) -> Result<FxHashMap<TxHash, StakeDoc>, StateError> {
    let mut accum = FxHashMap::default();
    for tx in txx {
        if tx.kind == TxKind::Stake {
            // first we check that the data is correct
            let stake_doc: StakeDoc =
                stdcode::deserialize(&tx.data).map_err(|_| StateError::MalformedTx)?;
            let curr_epoch = this.height.epoch();
            // then we check that the first coin is valid
            let first_coin = tx.outputs.get(0).ok_or(StateError::MalformedTx)?;

            let is_first_coin_not_a_sym: bool = first_coin.denom != Denom::Sym;

            // Are we operating under OLD BUGGY RULES?
            if (this.network == NetID::Mainnet || this.network == NetID::Testnet)
                && this.height.0 < 500000
            {
                log::warn!("LETTING THROUGH BAD STAKING TRANSACTION UNDER OLD BUGGY RULES");
                continue;
            }

            if is_first_coin_not_a_sym {
                return Err(StateError::MalformedTx);
            // then we check consistency
            } else if stake_doc.e_start > curr_epoch
                && stake_doc.e_post_end > stake_doc.e_start
                && stake_doc.syms_staked == first_coin.value
            {
                accum.insert(tx.hash_nosigs(), stake_doc);
            } else {
                log::warn!("**** REJECTING STAKER {:?} ****", stake_doc);
                continue;
            }
        }
    }
    Ok(accum)
}

fn check_tx_validity<C: ContentAddrStore>(
    this: &State<C>,
    tx: &Transaction,
    relevant_coins: &FxHashMap<CoinID, CoinDataHeight>,
    new_stakes: &FxHashMap<TxHash, StakeDoc>,
) -> Result<(), StateError> {
    let txhash = tx.hash_nosigs();
    let start = Instant::now();
    let scripts = tx.covenants_as_map();
    // build a map of input coins
    let mut in_coins: FxHashMap<Denom, u128> = FxHashMap::default();
    // get last header
    let last_header = this
        .history
        .get(&(this.height.0.saturating_sub(1).into()))
        .0
        .unwrap_or_else(|| this.clone().seal(None).header());
    // iterate through the inputs
    let mut good_scripts: FxHashSet<Address> = FxHashSet::default();
    for (spend_idx, coin_id) in tx.inputs.iter().enumerate() {
        if (new_stakes.contains_key(&coin_id.txhash)
            || this.stakes.get(&coin_id.txhash).0.is_some())
            && !((this.network == NetID::Mainnet || this.network == NetID::Testnet)
                && this.height.0 < 900000)
        // Workaround for BUGGY old code!
        {
            return Err(StateError::CoinLocked);
        }
        let coin_data = relevant_coins.get(coin_id);
        match coin_data {
            None => return Err(StateError::NonexistentCoin(*coin_id)),
            Some(coin_data) => {
                log::trace!(
                    "coin_data {:?} => {:?} for txid {:?}",
                    coin_id,
                    coin_data,
                    tx.hash_nosigs()
                );
                if !good_scripts.contains(&coin_data.coin_data.covhash) {
                    let script = Covenant(
                        scripts
                            .get(&coin_data.coin_data.covhash)
                            .ok_or(StateError::NonexistentScript(coin_data.coin_data.covhash))?
                            .clone(),
                    );
                    if !script.check(
                        tx,
                        CovenantEnv {
                            parent_coinid: *coin_id,
                            parent_cdh: coin_data.clone(),
                            spender_index: spend_idx as u8,
                            last_header,
                        },
                    ) {
                        return Err(StateError::ViolatesScript(coin_data.coin_data.covhash));
                    }
                    good_scripts.insert(coin_data.coin_data.covhash);
                }
                in_coins.insert(
                    coin_data.coin_data.denom,
                    in_coins.get(&coin_data.coin_data.denom).unwrap_or(&0)
                        + coin_data.coin_data.value.0,
                );
            }
        }
    }
    log::trace!("{}: processed all inputs {:?}", txhash, start.elapsed());
    // balance inputs and outputs. ignore outputs with empty cointype (they create a new token kind)
    let out_coins = tx.total_outputs();
    if tx.kind != TxKind::Faucet {
        for (currency, value) in out_coins.iter() {
            // we skip the created doscs for a DoscMint transaction, which are left for later.
            if *currency == Denom::NewCoin
                || (tx.kind == TxKind::DoscMint && *currency == Denom::Erg)
            {
                continue;
            }
            let in_value = if let Some(in_value) = in_coins.get(currency) {
                *in_value
            } else {
                return Err(StateError::UnbalancedInOut);
            };
            if *value != CoinValue(in_value) {
                eprintln!(
                    "unbalanced: {} {:?} in, {} {:?} out",
                    CoinValue(in_value),
                    currency,
                    value,
                    currency
                );
                return Err(StateError::UnbalancedInOut);
            }
        }
    }

    Ok(())
}

/// Checks a doscmint transaction, and returns the implicit speed.
fn check_doscmint_validity<C: ContentAddrStore>(
    this: &State<C>,
    relevant_coins: &FxHashMap<CoinID, CoinDataHeight>,
    tx: &Transaction,
) -> Result<u128, StateError> {
    let coin_id = *tx.inputs.get(0).unwrap();
    let coin_data = relevant_coins
        .get(&coin_id)
        .ok_or(StateError::NonexistentCoin(coin_id))?;
    // make sure the time is long enough that we can easily measure it
    if (this.height - coin_data.height).0 < 100 && this.network == NetID::Mainnet {
        log::warn!("rejecting doscmint due to too recent");
        return Err(StateError::InvalidMelPoW);
    }
    // construct puzzle seed
    let chi = tmelcrypt::hash_keyed(
        &this
            .history
            .get(&coin_data.height)
            .0
            .ok_or(StateError::InvalidMelPoW)?
            .hash(),
        &stdcode::serialize(tx.inputs.get(0).unwrap()).unwrap(),
    );
    // get difficulty and proof
    let (difficulty, proof_bytes): (u32, Vec<u8>) =
        stdcode::deserialize(&tx.data).map_err(|e| {
            log::warn!("rejecting doscmint due to malformed proof: {:?}", e);
            StateError::InvalidMelPoW
        })?;
    let proof = melpow::Proof::from_bytes(&proof_bytes).unwrap();

    // try verifying the proof under the old and the new system
    let is_tip910 = {
        if proof.verify(&chi, difficulty as _, LegacyMelPowHash) {
            false
        } else if proof.verify(&chi, difficulty as _, Tip910MelPowHash) {
            true
        } else {
            return Err(StateError::InvalidMelPoW);
        }
    };

    // compute speeds
    let my_speed = if is_tip910 { 100 } else { 1 } * 2u128.pow(difficulty)
        / (this.height - coin_data.height).0 as u128;
    let reward_real = melmint::calculate_reward(
        my_speed,
        this.history
            .get(&BlockHeight(this.height.0 - 1))
            .0
            .ok_or(StateError::InvalidMelPoW)?
            .dosc_speed,
        difficulty,
        is_tip910,
    );

    let reward_nom = CoinValue(melmint::dosc_to_erg(this.height, reward_real));
    // ensure that the total output of DOSCs is correct
    let total_dosc_output = tx
        .total_outputs()
        .get(&Denom::Erg)
        .cloned()
        .unwrap_or_default();
    if total_dosc_output > reward_nom {
        return Err(StateError::InvalidMelPoW);
    }
    Ok(my_speed)
}

fn faucet_dedup_pseudocoin(txhash: TxHash) -> CoinID {
    CoinID {
        txhash: tmelcrypt::hash_keyed(b"fdp", &txhash.0).into(),
        index: 0,
    }
}
