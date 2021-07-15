use num::{integer::Roots, rational::Ratio, BigInt, BigRational};
use std::{cell::RefCell, convert::TryInto};
use tap::Pipe;

use super::melswap::PoolState;
use crate::{
    CoinData, CoinDataHeight, Denom, PoolKey, State, Transaction, TxKind, MAX_COINVAL,
    MICRO_CONVERTER, TIP_902_HEIGHT,
};

thread_local! {
    static INFLATOR_TABLE: RefCell<Vec<u128>> = Default::default();
}

/// Internal DOSC inflator. Returns how many µNomDOSC is 1 DOSC.
fn micronomdosc_per_dosc(height: u64) -> u128 {
    // fn inner(height: u64) -> u128 {
    //     if height == 0 {
    //         MICRO_CONVERTER
    //     } else {
    //         // HACK: "segmented stacks"
    //         let last = stacker::maybe_grow(32 * 1024, 1024 * 1024, || inner(height - 1));
    //         (last + 1).max(last + last / 2_000_000)
    //     }
    // }
    INFLATOR_TABLE.with(|dpt| {
        let mut dpt = dpt.borrow_mut();
        if dpt.is_empty() {
            dpt.push(MICRO_CONVERTER);
        }
        while dpt.len() < (height + 1) as usize {
            let last = dpt.last().copied().unwrap();
            dpt.push((last + 1).max(last + last / 2_000_000));
        }

        dpt[height as usize]
    })
}

/// DOSC inflation ratio.
pub fn dosc_inflator(height: u64) -> BigRational {
    BigRational::from((
        BigInt::from(micronomdosc_per_dosc(height)),
        BigInt::from(MICRO_CONVERTER),
    ))
}

/// DOSC inflation calculator.
pub fn dosc_inflate_r2n(height: u64, real: u128) -> u128 {
    let ratio = dosc_inflator(height);
    let result = ratio * BigRational::from(BigInt::from(real));
    result
        .floor()
        .numer()
        .to_biguint()
        .unwrap()
        .try_into()
        .expect("dosc inflated so much it doesn't fit into a u128")
}

/// Reward calculator. Returns the value in real DOSC.
pub fn calculate_reward(my_speed: u128, dosc_speed: u128, difficulty: u32) -> u128 {
    let work_done = 2u128.pow(difficulty as _);
    // correct calculation with bigints
    let result = (BigInt::from(work_done) * BigInt::from(my_speed) * BigInt::from(MICRO_CONVERTER))
        / (BigInt::from(dosc_speed).pow(2) * BigInt::from(2880));
    result.try_into().unwrap_or(u128::MAX)
}

/// Presealing function that is called before a state is sealed to apply melmint actions.
pub fn preseal_melmint(state: State) -> State {
    let state = create_builtins(state);
    assert!(state.pools.val_iter().count() >= 2);
    let state = process_swaps(state);
    assert!(state.pools.val_iter().count() >= 2);
    let state = process_deposits(state);
    assert!(state.pools.val_iter().count() >= 2);
    let state = process_withdrawals(state);
    assert!(state.pools.val_iter().count() >= 2);
    process_pegging(state)
}

/// Creates the built-in pools if they don't exist. The built-in pools start out with nonzero liq, so that they can never be completely depleted. This ensures that built-in pools will always exist in the state.
fn create_builtins(mut state: State) -> State {
    let mut def = PoolState::new_empty();
    let _ = def.deposit(MICRO_CONVERTER * 1000, MICRO_CONVERTER * 1000);
    if state.pools.get(&PoolKey::mel_and(Denom::Sym)).0.is_none() {
        state.pools.insert(PoolKey::mel_and(Denom::Sym), def)
    }
    if state
        .pools
        .get(&PoolKey::mel_and(Denom::NomDosc))
        .0
        .is_none()
    {
        state.pools.insert(PoolKey::mel_and(Denom::NomDosc), def)
    }
    if state.height >= TIP_902_HEIGHT
        && state
            .pools
            .get(&PoolKey::new(Denom::NomDosc, Denom::Sym))
            .0
            .is_none()
    {
        state
            .pools
            .insert(PoolKey::new(Denom::NomDosc, Denom::Sym), def)
    }
    state
}

/// Process swaps.
fn process_swaps(mut state: State) -> State {
    // find the swap requests
    let swap_reqs: Vec<Transaction> = state
        .transactions
        .val_iter()
        .filter_map(|tx| {
            (!tx.outputs.is_empty()).then(|| ())?; // ensure not empty
            state.coins.get(&tx.output_coinid(0)).0?; // ensure that first output is unspent
            let pool_key = PoolKey::from_bytes(&tx.data)?; // ensure that data contains a pool key
            state.pools.get(&pool_key).0?; // ensure that pool key points to a valid pool
            (tx.outputs[0].denom == pool_key.left || tx.outputs[0].denom == pool_key.right)
                .then(|| ())?; // ensure that the first output is either left or right
            Some(tx)
        })
        .collect::<Vec<Transaction>>();
    log::trace!("{} swap requests", swap_reqs.len());
    // find the pools mentioned
    let mut pools = swap_reqs
        .iter()
        .map(|tx| PoolKey::from_bytes(&tx.data).expect("swap_reqs contains invalid poolkey"))
        .collect::<Vec<PoolKey>>();

    pools.sort_unstable();
    pools.dedup();
    // for each pool
    for pool in pools {
        let relevant_swaps: Vec<Transaction> = swap_reqs
            .iter()
            .filter(|tx| Some(pool) == PoolKey::from_bytes(&tx.data))
            .cloned()
            .collect();
        log::trace!(
            "{} relevant swaps for pool {:?}",
            relevant_swaps.len(),
            pool
        );
        let mut pool_state = state.pools.get(&pool).0.unwrap();
        // sum up total lefts and rights
        let total_lefts = relevant_swaps
            .iter()
            .map(|tx| {
                if tx.outputs[0].denom == pool.left {
                    tx.outputs[0].value
                } else {
                    0
                }
            })
            .fold(0u128, |a, b| a.saturating_add(b));
        let total_rights = relevant_swaps
            .iter()
            .map(|tx| {
                if tx.outputs[0].denom == pool.right {
                    tx.outputs[0].value
                } else {
                    0
                }
            })
            .fold(0u128, |a, b| a.saturating_add(b));
        // transmute coins
        let (left_withdrawn, right_withdrawn) = pool_state.swap_many(total_lefts, total_rights);

        for mut swap in relevant_swaps {
            let correct_coinid = swap.output_coinid(0);

            if swap.outputs[0].denom == pool.left {
                swap.outputs[0].denom = pool.right;
                swap.outputs[0].value = multiply_frac(
                    right_withdrawn,
                    Ratio::new(swap.outputs[0].value, total_lefts),
                )
                .min(MAX_COINVAL);
            } else {
                swap.outputs[0].denom = pool.left;
                swap.outputs[0].value = multiply_frac(
                    left_withdrawn,
                    Ratio::new(swap.outputs[0].value, total_rights),
                )
                .min(MAX_COINVAL);
            }
            state.coins.insert(
                correct_coinid,
                CoinDataHeight {
                    coin_data: swap.outputs[0].clone(),
                    height: state.height,
                },
            );
        }
        state.pools.insert(pool, pool_state);
    }

    state
}

/// Process deposits.
fn process_deposits(mut state: State) -> State {
    // find the deposit requests
    let deposit_reqs = state
        .transactions
        .val_iter()
        .filter_map(|tx| {
            (tx.kind == TxKind::LiqDeposit
                && tx.outputs.len() >= 2
                && state.coins.get(&tx.output_coinid(0)).0.is_some()
                && state.coins.get(&tx.output_coinid(1)).0.is_some())
            .then(|| ())?;
            let pool_key = PoolKey::from_bytes(&tx.data)?;
            (tx.outputs[0].denom == pool_key.left && tx.outputs[1].denom == pool_key.right)
                .then(|| tx)
        })
        .collect::<Vec<_>>();
    log::trace!("{} deposit reqs", deposit_reqs.len());
    // find the pools mentioned
    let pools = deposit_reqs
        .iter()
        .filter_map(|tx| PoolKey::from_bytes(&tx.data))
        .collect::<Vec<_>>()
        .pipe(|mut v| {
            v.sort();
            v.dedup();
            v
        });
    for pool in pools {
        let relevant_txx: Vec<Transaction> = deposit_reqs
            .iter()
            .filter(|tx| PoolKey::from_bytes(&tx.data) == Some(pool))
            .cloned()
            .collect();
        // sum up total lefts and rights
        let total_lefts: u128 = relevant_txx
            .iter()
            .map(|tx| tx.outputs[0].value)
            .fold(0u128, |a, b| a.saturating_add(b));
        let total_rights: u128 = relevant_txx
            .iter()
            .map(|tx| tx.outputs[1].value)
            .fold(0u128, |a, b| a.saturating_add(b));
        let total_mtsqrt = total_lefts.sqrt().saturating_mul(total_rights.sqrt());
        // main logic here
        let total_liqs = if let Some(mut pool_state) = state.pools.get(&pool).0 {
            let liq = pool_state.deposit(total_lefts, total_rights);
            state.pools.insert(pool, pool_state);
            liq
        } else {
            let mut pool_state = PoolState::new_empty();
            let liq = pool_state.deposit(total_lefts, total_rights);
            state.pools.insert(pool, pool_state);
            liq
        };
        // divvy up the liqs
        for mut deposit in relevant_txx {
            let correct_coinid = deposit.output_coinid(0);
            let my_mtsqrt = deposit.outputs[0]
                .value
                .sqrt()
                .saturating_mul(deposit.outputs[1].value.sqrt());
            deposit.outputs[0].denom = pool.liq_token_denom();
            deposit.outputs[0].value =
                multiply_frac(total_liqs, Ratio::new(my_mtsqrt, total_mtsqrt));
            state.coins.insert(
                correct_coinid,
                CoinDataHeight {
                    coin_data: deposit.outputs[0].clone(),
                    height: state.height,
                },
            );
            state.coins.delete(&deposit.output_coinid(1));
        }
    }
    state
}

/// Process deposits.
fn process_withdrawals(mut state: State) -> State {
    // find the withdrawal requests
    let withdraw_reqs: Vec<Transaction> = state
        .transactions
        .val_iter()
        .filter_map(|tx| {
            (tx.kind == TxKind::LiqWithdraw
                && tx.outputs.len() == 1
                && state.coins.get(&tx.output_coinid(0)).0.is_some())
            .then(|| ())?;
            let pool_key = PoolKey::from_bytes(&tx.data)?;
            state.pools.get(&pool_key).0?;
            (tx.outputs[0].denom == pool_key.liq_token_denom()).then(|| tx)
        })
        .collect::<Vec<Transaction>>();
    // find the pools mentioned
    let pools = withdraw_reqs
        .iter()
        .filter_map(|tx| PoolKey::from_bytes(&tx.data))
        .collect::<Vec<_>>()
        .pipe(|mut v| {
            v.sort();
            v.dedup();
            v
        });
    for pool in pools {
        let relevant_txx: Vec<Transaction> = withdraw_reqs
            .iter()
            .filter(|tx| PoolKey::from_bytes(&tx.data) == Some(pool))
            .cloned()
            .collect();
        // sum up total liqs
        let total_liqs = relevant_txx
            .iter()
            .map(|tx| tx.outputs[0].value)
            .fold(0u128, |a, b| a.saturating_add(b));
        // get the state
        let mut pool_state = state.pools.get(&pool).0.unwrap();
        let (total_left, total_write) = pool_state.withdraw(total_liqs);
        state.pools.insert(pool, pool_state);
        // divvy up the lefts and rights
        for mut deposit in relevant_txx {
            let coinid_0 = deposit.output_coinid(0);
            let coinid_1 = deposit.output_coinid(1);

            let my_liqs = deposit.outputs[0].value;
            deposit.outputs[0].denom = pool.left;
            deposit.outputs[0].value = multiply_frac(total_left, Ratio::new(my_liqs, total_liqs));
            let synth = CoinData {
                denom: pool.right,
                value: multiply_frac(total_write, Ratio::new(my_liqs, total_liqs)),
                covhash: deposit.outputs[0].covhash,
                additional_data: deposit.outputs[0].additional_data.clone(),
            };

            state.coins.insert(
                coinid_0,
                CoinDataHeight {
                    coin_data: deposit.outputs[0].clone(),
                    height: state.height,
                },
            );
            state.coins.insert(
                coinid_1,
                CoinDataHeight {
                    coin_data: synth,
                    height: state.height,
                },
            );
        }
    }
    state
}

/// Process pegging.
fn process_pegging(mut state: State) -> State {
    // first calculate the implied sym/nomDOSC exchange rate
    let x_sd = if state.height >= TIP_902_HEIGHT {
        state
            .pools
            .get(&PoolKey::new(Denom::Sym, Denom::NomDosc))
            .0
            .unwrap()
            .implied_price() // doscs per sym
            .recip() // syms per dosc
    } else {
        let x_s = state
            .pools
            .get(&PoolKey::mel_and(Denom::Sym))
            .0
            .unwrap()
            .implied_price()
            .recip();
        let x_d = state
            .pools
            .get(&PoolKey::mel_and(Denom::NomDosc))
            .0
            .unwrap()
            .implied_price()
            .recip();
        x_s / x_d
    };

    let throttler = if state.height >= TIP_902_HEIGHT {
        200
    } else {
        1000
    };

    // get the right pool
    let mut sm_pool = state.pools.get(&PoolKey::mel_and(Denom::Sym)).0.unwrap();
    let konstant = BigInt::from(sm_pool.lefts) * BigInt::from(sm_pool.rights);
    // desired mel and sym
    let desired_x_sm = dosc_inflator(state.height) * x_sd;
    let desired_mel_sqr = BigRational::from(konstant.clone()) / desired_x_sm.clone();
    let desired_mel: u128 = desired_mel_sqr
        .floor()
        .numer()
        .sqrt()
        .try_into()
        .unwrap_or(u128::MAX);
    let desired_sym_sqr = BigRational::from(konstant) * desired_x_sm;
    let desired_sym: u128 = desired_sym_sqr
        .floor()
        .numer()
        .sqrt()
        .try_into()
        .unwrap_or(u128::MAX);
    // we nudge towards the desired level entirely through "normal" operations
    if desired_mel > sm_pool.lefts {
        let delta = (desired_mel - sm_pool.lefts) / throttler;
        // we increase mel liquidity by delta, throwing away the syms generated.
        // this nudges the exchange rate while minimizing long-term inflation
        let _ = sm_pool.swap_many(delta, 0);
    }
    if desired_sym > sm_pool.rights {
        let delta = (desired_sym - sm_pool.rights) / throttler;
        let _ = sm_pool.swap_many(0, delta);
    }
    state.pools.insert(PoolKey::mel_and(Denom::Sym), sm_pool);
    // return the state now
    assert!(state.pools.val_iter().count() >= 2);
    state
}

fn multiply_frac(x: u128, frac: Ratio<u128>) -> u128 {
    let frac = Ratio::new(BigInt::from(*frac.numer()), BigInt::from(*frac.denom()));
    let result = BigRational::from(BigInt::from(x)) * frac;
    result.floor().numer().try_into().unwrap_or(u128::MAX)
}

#[cfg(test)]
mod tests {
    use crate::{
        melvm,
        testing::fixtures::{genesis_mel_coin_id, genesis_state},
        CoinID, Denom,
    };

    use super::*;

    #[test]
    fn math() {
        assert_eq!(multiply_frac(1000, Ratio::new(2, 1)), 2000)
    }

    #[test]
    // test a simple deposit flow
    fn simple_deposit() {
        let (my_pk, my_sk) = tmelcrypt::ed25519_keygen();
        let my_covhash = melvm::Covenant::std_ed25519_pk_legacy(my_pk).hash();
        let start_state = genesis_state(
            CoinID::zero_zero(),
            CoinDataHeight {
                coin_data: CoinData {
                    value: (1 << 64) + 4000000,
                    denom: Denom::Mel,
                    covhash: my_covhash,
                    additional_data: vec![],
                },
                height: 100,
            },
            Default::default(),
        );
        // test sealing
        let mut second_state = start_state.seal(None).next_state();
        // deposit the genesis as a custom-token pool
        let newcoin_tx = Transaction {
            kind: TxKind::Normal,
            inputs: vec![genesis_mel_coin_id()],
            outputs: vec![
                CoinData {
                    covhash: my_covhash,
                    value: (1 << 64) + 2000000,
                    denom: Denom::Mel,
                    additional_data: vec![],
                },
                CoinData {
                    covhash: my_covhash,
                    value: 1 << 64,
                    denom: Denom::NewCoin,
                    additional_data: vec![],
                },
            ],
            fee: 2000000,
            scripts: vec![melvm::Covenant::std_ed25519_pk_legacy(my_pk)],
            data: vec![],
            sigs: vec![],
        }
        .signed_ed25519(my_sk);
        second_state.apply_tx(&newcoin_tx).unwrap();
        let pool_key = PoolKey::mel_and(Denom::Custom(newcoin_tx.hash_nosigs()));
        let deposit_tx = Transaction {
            kind: TxKind::LiqDeposit,
            inputs: vec![newcoin_tx.output_coinid(0), newcoin_tx.output_coinid(1)],
            outputs: vec![
                CoinData {
                    covhash: my_covhash,
                    value: (1 << 64),
                    denom: pool_key.left,
                    additional_data: vec![],
                },
                CoinData {
                    covhash: my_covhash,
                    value: 1 << 64,
                    denom: pool_key.right,
                    additional_data: vec![],
                },
            ],
            fee: 2000000,
            scripts: vec![melvm::Covenant::std_ed25519_pk_legacy(my_pk)],
            data: pool_key.to_bytes(), // this is important, since it "points" to the pool
            sigs: vec![],
        }
        .signed_ed25519(my_sk);
        second_state.apply_tx(&deposit_tx).unwrap();
        let second_sealed = second_state.seal(None);
        for pool in second_sealed.inner_ref().pools.val_iter() {
            dbg!(pool);
        }
        dbg!(second_sealed.inner_ref().pools.get(&pool_key).0.unwrap());
    }
}