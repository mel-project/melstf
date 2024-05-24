use std::{collections::HashMap, hash::BuildHasherDefault, time::Instant};

use bytes::Bytes;
use melpow::Proof;
use melstructs::{
    Address, BlockHeight, CoinData, CoinDataHeight, CoinID, CoinValue, Denom, Header, NetID,
    StakeDoc, Transaction, TxHash, TxKind,
};
use melvm::{covenant_weight_from_bytes, Covenant, CovenantEnv};
use novasmt::ContentAddrStore;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use tmelcrypt::HashVal;

use crate::{melmint, LegacyMelPowHash, StateError, Tip910MelPowHash, UnsealedState};

// TODO: add proper description of this exploit
const INFLATION_BUG_TX_HASH: &str =
    "30a60b20830f000f755b70c57c998553a303cc11f8b1f574d5e9f7e26b645d8b";

/// Applies a batch of transactions to the state and returns the new state afterwards.
pub fn apply_tx_batch_impl<C: ContentAddrStore>(
    this: &UnsealedState<C>,
    txx: &[Transaction],
) -> Result<UnsealedState<C>, StateError> {
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
                let new_speed = validate_and_get_doscmint_speed(this, &relevant_coins, tx)?;
                Ok(a.max(new_speed))
            },
        )
        .try_reduce(|| this.dosc_speed, |a, b| Ok(a.max(b)))?;

    // great, now we create the new state from the transactions
    let mut next_state = create_next_state(this.clone(), txx, &relevant_coins)?;

    // dosc
    next_state.dosc_speed = new_max_speed;

    // apply stakes
    for (k, v) in new_stakes {
        next_state.stakes.add_stake(k, v);
    }
    Ok(next_state)
}

fn handle_faucet_tx<C: ContentAddrStore>(
    state: &mut UnsealedState<C>,
    tx: &Transaction,
) -> Result<(), StateError> {
    // dedup faucet
    if tx.kind == TxKind::Faucet {
        let bug_compatible_with_inflation_exploit =
            tx.hash_nosigs().to_string() == INFLATION_BUG_TX_HASH;
        // exception to be bug-compatible with the one guy who exploited the inflation bug
        if state.network == NetID::Mainnet && !bug_compatible_with_inflation_exploit {
            log::error!(
                "rejecting mainnet faucet with hash {:?}",
                tx.hash_nosigs().to_string()
            );
            return Err(StateError::MalformedTx);
        }
        if bug_compatible_with_inflation_exploit {
            log::error!(
                "allowing mainnet faucet with hash {:?}",
                tx.hash_nosigs().to_string()
            );
        }

        let pseudocoin = faucet_dedup_pseudocoin(tx.hash_nosigs());
        if state.coins.get_coin(pseudocoin).is_some() {
            return Err(StateError::DuplicateTx);
        }
        // We do not insert the pseudocoin if this transaction is the one transaction that was buggy, in block 1214212, on the mainnet.
        // Back then, we were buggy in two ways: we allowed mainnet faucets accidentally, and we didn't insert the dedup pseudocoin properly!
        if !bug_compatible_with_inflation_exploit {
            state.coins.insert_coin(
                pseudocoin,
                CoinDataHeight {
                    coin_data: CoinData {
                        denom: Denom::Mel,
                        value: 0.into(),
                        additional_data: vec![].into(),
                        covhash: HashVal::default().into(),
                    },
                    height: 0.into(),
                },
            );
        }
    }

    Ok(())
}

fn create_next_state<C: ContentAddrStore>(
    mut next_state: UnsealedState<C>,
    transactions: &[Transaction],
    relevant_coins: &FxHashMap<CoinID, CoinDataHeight>,
) -> Result<UnsealedState<C>, StateError> {
    for tx in transactions {
        let txhash = tx.hash_nosigs();

        if tx.kind == TxKind::Faucet {
            handle_faucet_tx(&mut next_state, tx)?;
        }

        for (i, _) in tx.outputs.iter().enumerate() {
            let coinid = CoinID::new(txhash, i as u8);
            // this filters out coins that we get rid of (e.g. due to them going to the coin destruction cov)
            if let Some(coin_data) = relevant_coins.get(&coinid) {
                next_state.coins.insert_coin(coinid, coin_data.clone());
            }
        }
        for coinid in tx.inputs.iter() {
            next_state.coins.remove_coin(*coinid);
        }

        // fees
        let min_fee = tx.base_fee(next_state.fee_multiplier, 0, |c| {
            covenant_weight_from_bytes(c)
        });
        if tx.fee < min_fee {
            return Err(StateError::InsufficientFees(min_fee));
        } else {
            let tips = tx.fee - min_fee;
            next_state.tips.0 = next_state.tips.0.saturating_add(tips.0);
            next_state.fee_pool.0 = next_state.fee_pool.0.saturating_add(min_fee.0);
        }
        next_state.transactions.insert(tx.clone());
    }
    Ok(next_state)
}

/// This collects all input and output coins referenced by the given transactions while filtering out any coins set to be destroyed.
/// This iterates over the given transactions and:
/// - does some light (incomplete) validation on the transaction
/// - collects the intput and output coins referenced by the state
///
/// NOTE: Coins specified in a transaction's `output` that have a `Address::coin_destroy()` covhash are permanently destroyed.
fn load_relevant_coins<C: ContentAddrStore>(
    this: &UnsealedState<C>,
    txx: &[Transaction],
) -> Result<FxHashMap<CoinID, CoinDataHeight>, StateError> {
    let mut accum: FxHashMap<CoinID, CoinDataHeight> = FxHashMap::default();

    // add the ones created in this batch
    for tx in txx {
        if !tx.is_well_formed() {
            return Err(StateError::MalformedTx);
        }

        let coins_to_add = output_coins_from_tx(tx, this.height);
        if !coins_to_add.is_empty() {
            accum.extend(coins_to_add);
        }
    }

    let input_coins = extract_input_coins(txx, this, &accum)?;
    accum.extend(input_coins);

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

fn extract_input_coins<C: ContentAddrStore>(
    transactions: &[Transaction],
    state: &UnsealedState<C>,
    coins_so_far: &FxHashMap<CoinID, CoinDataHeight>,
) -> Result<FxHashMap<CoinID, CoinDataHeight>, StateError> {
    let mut accum: FxHashMap<CoinID, CoinDataHeight> = FxHashMap::default();

    // add the ones *referenced* in this batch
    let cache: FxHashMap<CoinID, Option<CoinDataHeight>> = transactions
        .into_par_iter()
        .flat_map(|transaction| transaction.inputs.par_iter())
        .map(|input| (*input, state.coins.get_coin(*input)))
        .collect();

    for tx in transactions {
        for input in tx.inputs.iter() {
            if !coins_so_far.contains_key(input) {
                let from_disk = cache
                    .get(input)
                    .unwrap()
                    .clone()
                    .ok_or(StateError::NonexistentCoin(*input))?;
                accum.insert(*input, from_disk);
            }
        }
    }

    Ok(accum)
}

fn output_coins_from_tx(
    tx: &Transaction,
    height: BlockHeight,
) -> FxHashMap<CoinID, CoinDataHeight> {
    tx.outputs
        .par_iter()
        .enumerate()
        .filter_map(|(i, coin_data)| {
            let mut coin_data = coin_data.clone();
            if coin_data.denom == Denom::NewCustom {
                coin_data.denom = Denom::Custom(tx.hash_nosigs());
            }

            // if covenant hash is zero, this destroys the coins permanently
            if coin_data.covhash != Address::coin_destroy() {
                Some((
                    CoinID::new(tx.hash_nosigs(), i as u8),
                    CoinDataHeight { coin_data, height },
                ))
            } else {
                None
            }
        })
        .collect::<FxHashMap<CoinID, CoinDataHeight>>()
}

fn coin_is_denom(coin_data: &CoinData, denom: Denom) -> bool {
    coin_data.denom == denom
}

fn stake_is_consistent(stake_doc: &StakeDoc, curr_epoch: u64, coin: &CoinData) -> bool {
    stake_doc.e_start > curr_epoch
        && stake_doc.e_post_end > stake_doc.e_start
        && stake_doc.syms_staked == coin.value
}

fn load_stake_info<C: ContentAddrStore>(
    this: &UnsealedState<C>,
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
            if !coin_is_denom(first_coin, Denom::Sym) {
                return Err(StateError::MalformedTx);
            }

            if stake_is_consistent(&stake_doc, curr_epoch, first_coin) {
                accum.insert(tx.hash_nosigs(), stake_doc);
            } else {
                log::warn!("**** REJECTING STAKER {:?} ****", stake_doc);
                continue;
            }
        }
    }
    Ok(accum)
}

fn validate_tx_scripts(
    spend_idx: usize,
    coin_id: &CoinID,
    tx: &Transaction,
    coin_data: &CoinDataHeight,
    last_header: Header,
    scripts: HashMap<Address, Bytes>,
    good_scripts: &FxHashSet<Address>,
) -> Result<(), StateError> {
    log::trace!(
        "coin_data {:?} => {:?} for txid {:?}",
        coin_id,
        coin_data,
        tx.hash_nosigs()
    );
    if !good_scripts.contains(&coin_data.coin_data.covhash) {
        let script = Covenant::from_bytes(
            &scripts
                .get(&coin_data.coin_data.covhash)
                .ok_or(StateError::NonexistentScript(coin_data.coin_data.covhash))?
                .clone(),
        )
        .map_err(|_| StateError::MalformedTx)?;
        if !script
            .execute(
                tx,
                Some(CovenantEnv {
                    parent_coinid: *coin_id,
                    parent_cdh: coin_data.clone(),
                    spender_index: spend_idx as u8,
                    last_header,
                }),
            )
            .map(|v| v.into_bool())
            .unwrap_or(false)
        {
            return Err(StateError::ViolatesScript(coin_data.coin_data.covhash));
        }
    }

    Ok(())
}

/// Balance inputs and outputs. Ignore outputs with empty cointype (they create a new token kind).
fn check_tx_coins_balanced(
    tx_kind: TxKind,
    in_coins: HashMap<Denom, u128, BuildHasherDefault<FxHasher>>,
    out_coins: HashMap<Denom, CoinValue>,
) -> Result<(), StateError> {
    if tx_kind != TxKind::Faucet {
        for (currency, value) in out_coins.iter() {
            // we skip the created doscs for a DoscMint transaction, which are left for later.
            if *currency == Denom::NewCustom
                || (tx_kind == TxKind::DoscMint && *currency == Denom::Erg)
            {
                continue;
            }

            let in_value = if let Some(in_value) = in_coins.get(currency) {
                *in_value
            } else {
                log::debug!("UnbalancedInOut: No {currency}");
                return Err(StateError::UnbalancedInOut);
            };

            if *value != CoinValue(in_value) {
                log::debug!(
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

fn check_tx_validity<C: ContentAddrStore>(
    this: &UnsealedState<C>,
    tx: &Transaction,
    relevant_coins: &FxHashMap<CoinID, CoinDataHeight>,
    new_stakes: &FxHashMap<TxHash, StakeDoc>,
) -> Result<(), StateError> {
    let txhash = tx.hash_nosigs();
    let start = Instant::now();
    let scripts = tx.covenants_as_map();

    let mut in_coins: FxHashMap<Denom, u128> = FxHashMap::default();

    // get last header
    let last_header = this
        .history
        .get(&(this.height.0.saturating_sub(1).into()))
        .unwrap_or_else(|| this.clone().seal(None).header());

    let mut good_scripts: FxHashSet<Address> = FxHashSet::default();
    for (spend_idx, coin_id) in tx.inputs.iter().enumerate() {
        // Workaround for BUGGY old code!
        // TODO: add some details for this
        if (new_stakes.contains_key(&coin_id.txhash)
            || this.stakes.get_stake(coin_id.txhash).is_some())
            && !((this.network == NetID::Mainnet || this.network == NetID::Testnet)
                && this.height.0 < 900000)
        {
            return Err(StateError::CoinLocked);
        }
        let coin_data = relevant_coins.get(coin_id);
        match coin_data {
            None => return Err(StateError::NonexistentCoin(*coin_id)),
            Some(coin_data) => {
                if !good_scripts.contains(&coin_data.coin_data.covhash) {
                    validate_tx_scripts(
                        spend_idx,
                        coin_id,
                        tx,
                        coin_data,
                        last_header,
                        scripts.clone(),
                        &good_scripts,
                    )?;

                    good_scripts.insert(coin_data.coin_data.covhash);
                }

                let amount = in_coins.get(&coin_data.coin_data.denom).unwrap_or(&0)
                    + coin_data.coin_data.value.0;
                in_coins.insert(coin_data.coin_data.denom, amount);
            }
        }
    }

    log::trace!("{}: processed all inputs {:?}", txhash, start.elapsed());
    let out_coins = tx.total_outputs();
    check_tx_coins_balanced(tx.kind, in_coins, out_coins)?;
    Ok(())
}

fn proof_is_tip910(proof: Proof, puzzle: &HashVal, difficulty: u32) -> Result<bool, StateError> {
    // try verifying the proof under the old and the new system
    if proof.verify(puzzle, difficulty as _, LegacyMelPowHash) {
        Ok(false)
    } else if proof.verify(puzzle, difficulty as _, Tip910MelPowHash) {
        Ok(true)
    } else {
        println!("proof_is_tip910() verification failed");
        Err(StateError::InvalidMelPoW)
    }
}

fn compute_doscmint_speed(
    is_tip910: bool,
    difficulty: u32,
    state_height: BlockHeight,
    coin_height: BlockHeight,
) -> u128 {
    (if is_tip910 { 100 } else { 1 }) * 2u128.pow(difficulty)
        / (state_height - coin_height).0 as u128
}

/// Ensure that the total output of DOSCs is correct.
fn check_dosc_total_output(tx: &Transaction, reward_nom: CoinValue) -> Result<(), StateError> {
    let total_dosc_output = tx
        .total_outputs()
        .get(&Denom::Erg)
        .cloned()
        .unwrap_or_default();
    println!("claimed={total_dosc_output}, actual={reward_nom}");
    println!("OUTPUTS: {:?}", tx.outputs);
    if total_dosc_output > reward_nom {
        return Err(StateError::InvalidMelPoW);
    }
    Ok(())
}

/// Checks a doscmint transaction, and returns the implicit speed.
fn validate_and_get_doscmint_speed<C: ContentAddrStore>(
    this: &UnsealedState<C>,
    relevant_coins: &FxHashMap<CoinID, CoinDataHeight>,
    tx: &Transaction,
) -> Result<u128, StateError> {
    let coin_id = *tx.inputs.get(0).expect("this cannot happen, since by the time we get here we've checked that transactions have inputs");
    let coin_data = relevant_coins
        .get(&coin_id)
        .ok_or(StateError::NonexistentCoin(coin_id))?;
    println!("COIN_DATA.HEIGHTTTTTTTTTTTTTT = {}", coin_data.height);
    // make sure the time is long enough that we can easily measure it
    if (this.height - coin_data.height).0 < 100 && this.network == NetID::Mainnet {
        log::warn!("rejecting doscmint due to too recent");
        return Err(StateError::InvalidMelPoW);
    }
    // construct puzzle seed
    let puzzle = tmelcrypt::hash_keyed(
        this.history
            .get(&coin_data.height)
            .ok_or(StateError::InvalidMelPoW)?
            .hash(),
        &stdcode::serialize(tx.inputs.get(0).unwrap()).unwrap(),
    );
    // get difficulty and proof
    let (difficulty, proof_bytes): (u32, Vec<u8>) =
        stdcode::deserialize(&tx.data).map_err(|e| {
            log::warn!("rejecting doscmint due to malformed proof: {:?}", e);
            println!("malformed proof!");
            StateError::InvalidMelPoW
        })?;
    let proof = match melpow::Proof::from_bytes(&proof_bytes) {
        Some(p) => p,
        None => {
            log::warn!(
                "failed to deserialize MEL PoW proof from transaction {:?}",
                tx
            );
            return Err(StateError::MalformedTx);
        }
    };
    let is_tip910 = proof_is_tip910(proof, &puzzle, difficulty)?;

    let my_speed = compute_doscmint_speed(is_tip910, difficulty, this.height, coin_data.height);
    let reward_real = melmint::calculate_reward(
        my_speed,
        this.history
            .get(&BlockHeight(this.height.0 - 1))
            .ok_or(StateError::InvalidMelPoW)?
            .dosc_speed,
        difficulty,
        is_tip910,
    );

    let reward_nom = CoinValue(melmint::dosc_to_erg(this.height, reward_real));
    check_dosc_total_output(tx, reward_nom)?;
    Ok(my_speed)
}

pub(crate) fn faucet_dedup_pseudocoin(txhash: TxHash) -> CoinID {
    CoinID {
        txhash: tmelcrypt::hash_keyed(b"fdp", txhash.0).into(),
        index: 0,
    }
}
