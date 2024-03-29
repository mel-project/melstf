use crate::UnsealedState;

use std::{cell::RefCell, convert::TryInto};

use melpow::HashFunction;
use melstructs::{
    BlockHeight, CoinData, CoinDataHeight, CoinValue, Denom, NetID, PoolKey, PoolState,
    Transaction, TxKind, MAX_COINVAL, MICRO_CONVERTER,
};
use novasmt::ContentAddrStore;
use num::{integer::Roots, rational::Ratio, BigInt, BigRational};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use tap::Pipe;

thread_local! {
    static INFLATOR_TABLE: RefCell<Vec<u128>> = Default::default();
}

/// Internal DOSC inflator. Returns how many µNomDOSC is 1 DOSC.
fn microergs_per_dosc(height: BlockHeight) -> u128 {
    static INFLATOR_TABLE: Lazy<RwLock<Vec<u128>>> = Lazy::new(Default::default);
    let lol = INFLATOR_TABLE.read().get(height.0 as usize).copied();
    lol.unwrap_or_else(|| {
        let mut tab = INFLATOR_TABLE.write();
        if tab.is_empty() {
            tab.push(MICRO_CONVERTER);
        }
        while tab.len() < (height.0 + 1) as usize {
            let last = tab.last().copied().unwrap();
            tab.push((last + 1).max(last + last / 2_000_000));
        }
        tab[height.0 as usize]
    })
}

/// Legacy MelPoW hasher. More details can be found [here](https://github.com/themeliolabs/themelio-node/issues/100).
pub struct LegacyMelPowHash;

impl HashFunction for LegacyMelPowHash {
    fn hash(&self, b: &[u8], k: &[u8]) -> melpow::SVec<u8> {
        melpow::SVec::from_slice(blake3::keyed_hash(blake3::hash(k).as_bytes(), b).as_bytes())
    }
}

/// New (TIP-910) MelPoW hasher. More details can be found in [this issue](https://github.com/themeliolabs/themelio-node/issues/100).
pub struct Tip910MelPowHash;

impl HashFunction for Tip910MelPowHash {
    fn hash(&self, b: &[u8], k: &[u8]) -> melpow::SVec<u8> {
        let mut res = blake3::keyed_hash(blake3::hash(k).as_bytes(), b);
        for _ in 0..99 {
            res = blake3::hash(res.as_bytes());
        }
        melpow::SVec::from_slice(res.as_bytes())
    }
}

/// DOSC inflation ratio.
pub fn dosc_inflator(height: BlockHeight) -> BigRational {
    BigRational::from((
        BigInt::from(microergs_per_dosc(height)),
        BigInt::from(MICRO_CONVERTER),
    ))
}

/// DOSC inflation calculator.
pub fn dosc_to_erg(height: BlockHeight, real: u128) -> u128 {
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
pub fn calculate_reward(my_speed: u128, dosc_speed: u128, difficulty: u32, tip910: bool) -> u128 {
    let work_done = 2u128.pow(difficulty as _);
    let work_done = if tip910 {
        work_done.saturating_mul(100)
    } else {
        work_done
    };
    // correct calculation with bigints
    let result = (BigInt::from(work_done) * BigInt::from(my_speed) * BigInt::from(MICRO_CONVERTER))
        / (BigInt::from(dosc_speed).pow(2) * BigInt::from(2880));
    result.try_into().unwrap_or(u128::MAX)
}

/// Presealing function that is called before a state is sealed to apply melmint actions.
///
/// Internally, this prepares the given state for sealing:
/// - create built-in nonzero liquidity pools if they don't exist already
/// - process any swap, deposit, or withdrawal requests for the state's pools. This consists of finding a subset of transactions and pools and applying swaps on them as needed.
/// - process pegging for MEL. This includes some nudging of inflation rates, etc. More details can be found in the [specifications](https://docs.themelio.org/specifications/tech-melmint/).
pub fn preseal_melmint<C: ContentAddrStore>(state: UnsealedState<C>) -> UnsealedState<C> {
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

fn extract_pool_keys_sorted(transactions: &mut [Transaction]) -> Vec<PoolKey> {
    transactions
        .iter()
        .filter_map(|tx| PoolKey::from_bytes(&tx.data))
        .collect::<Vec<_>>()
        .pipe(|mut v| {
            v.sort();
            v.dedup();
            v
        })
}

fn transactions_for_pool(transactions: &[Transaction], pool_key: &PoolKey) -> Vec<Transaction> {
    transactions
        .iter()
        .filter(|tx| Some(pool_key) == PoolKey::from_bytes(&tx.data).as_ref())
        .cloned()
        .collect()
}

/// Creates the built-in pools if they don't exist. The built-in pools start out with nonzero liq, so that they can never be completely depleted. This ensures that built-in pools will always exist in the state.
fn create_builtins<C: ContentAddrStore>(mut state: UnsealedState<C>) -> UnsealedState<C> {
    let mut def = PoolState::new_empty();
    let _ = def.deposit(MICRO_CONVERTER * 1000, MICRO_CONVERTER * 1000);
    if state
        .pools
        .get(&PoolKey::new(Denom::Mel, Denom::Sym))
        .is_none()
    {
        state
            .pools
            .insert(PoolKey::new(Denom::Mel, Denom::Sym), def)
    }
    if state
        .pools
        .get(&PoolKey::new(Denom::Mel, Denom::Erg))
        .is_none()
    {
        state
            .pools
            .insert(PoolKey::new(Denom::Mel, Denom::Erg), def)
    }
    if state.tip_902()
        && state
            .pools
            .get(&PoolKey::new(Denom::Erg, Denom::Sym))
            .is_none()
    {
        state
            .pools
            .insert(PoolKey::new(Denom::Erg, Denom::Sym), def)
    }
    state
}

fn process_swaps_for_single_pool<C: ContentAddrStore>(
    pool: &PoolKey,
    state: &mut UnsealedState<C>,
    swaps: &mut Vec<Transaction>,
) {
    log::trace!("{} relevant swaps for pool {:?}", swaps.len(), pool);
    let mut pool_state = state.pools.get(pool).unwrap();
    // sum up total lefts and rights
    let total_lefts = swaps
        .iter()
        .map(|tx| {
            if tx.outputs[0].denom == pool.left() {
                tx.outputs[0].value
            } else {
                CoinValue(0)
            }
        })
        .fold(0u128, |a, b| a.saturating_add(b.0));
    let total_rights = swaps
        .iter()
        .map(|tx| {
            if tx.outputs[0].denom == pool.right() {
                tx.outputs[0].value
            } else {
                CoinValue(0)
            }
        })
        .fold(0u128, |a, b| a.saturating_add(b.0));
    // transmute coins
    let (left_withdrawn, right_withdrawn) = pool_state.swap_many(total_lefts, total_rights);

    swaps.iter_mut().for_each(|swap| {
        let correct_coinid = swap.output_coinid(0);

        if swap.outputs[0].denom == pool.left() {
            swap.outputs[0].denom = pool.right();
            swap.outputs[0].value = CoinValue(multiply_frac(
                right_withdrawn,
                Ratio::new(swap.outputs[0].value.0, total_lefts),
            ))
            .min(MAX_COINVAL);
        } else {
            swap.outputs[0].denom = pool.left();
            swap.outputs[0].value = CoinValue(multiply_frac(
                left_withdrawn,
                Ratio::new(swap.outputs[0].value.0, total_rights),
            ))
            .min(MAX_COINVAL);
        }
        state.coins.insert_coin(
            correct_coinid,
            CoinDataHeight {
                coin_data: swap.outputs[0].clone(),
                height: state.height,
            },
            state.tip_906(),
        );
    });

    state.pools.insert(*pool, pool_state);
}

/// Extract the swap requests from the given `State`.
fn get_swap_transactions<C: ContentAddrStore>(state: &UnsealedState<C>) -> Vec<Transaction> {
    state
        .transactions
        .iter()
        .cloned()
        .filter_map(|tx| {
            (!tx.outputs.is_empty()).then_some(())?; // ensure not empty
            state.coins.get_coin(tx.output_coinid(0))?; // ensure that first output is unspent
            let pool_key = PoolKey::from_bytes(&tx.data)?; // ensure that data contains a pool key
            state.pools.get(&pool_key)?; // ensure that pool key points to a valid pool
            (tx.outputs[0].denom == pool_key.left() || tx.outputs[0].denom == pool_key.right())
                .then_some(())?; // ensure that the first output is either left or right
            Some(tx)
        })
        .collect::<Vec<Transaction>>()
}

/// Process swaps.
fn process_swaps<C: ContentAddrStore>(mut state: UnsealedState<C>) -> UnsealedState<C> {
    let mut swap_reqs: Vec<Transaction> = get_swap_transactions(&state);

    log::trace!("{} swap requests", swap_reqs.len());

    // find the pools mentioned
    let pools = extract_pool_keys_sorted(&mut swap_reqs);

    // for each pool
    pools.iter().for_each(|pool| {
        let mut relevant_swaps: Vec<Transaction> = transactions_for_pool(&swap_reqs, pool);
        process_swaps_for_single_pool(pool, &mut state, &mut relevant_swaps);
    });

    state
}

fn process_deposits_for_single_pool<C: ContentAddrStore>(
    pool: &PoolKey,
    state: &mut UnsealedState<C>,
    deposits: &mut [Transaction],
) {
    // sum up total lefts and rights
    let total_lefts: u128 = deposits
        .iter()
        .map(|tx| tx.outputs[0].value.0)
        .fold(0u128, |a, b| a.saturating_add(b));

    let total_rights: u128 = deposits
        .iter()
        .map(|tx| tx.outputs[1].value.0)
        .fold(0u128, |a, b| a.saturating_add(b));

    let total_mtsqrt = total_lefts.sqrt().saturating_mul(total_rights.sqrt());
    // main logic here
    let total_liqs = if let Some(mut pool_state) = state.pools.get(pool) {
        let liq = pool_state.deposit(total_lefts, total_rights);
        state.pools.insert(*pool, pool_state);
        liq
    } else {
        let mut pool_state = PoolState::new_empty();
        let liq = pool_state.deposit(total_lefts, total_rights);
        state.pools.insert(*pool, pool_state);
        liq
    };
    // divvy up the liqs
    deposits.iter_mut().for_each(|deposit| {
        let original_tx = deposit.clone();
        let my_mtsqrt = deposit.outputs[0]
            .value
            .0
            .sqrt()
            .saturating_mul(deposit.outputs[1].value.0.sqrt());
        deposit.outputs[0].denom = pool.liq_token_denom();
        deposit.outputs[0].value =
            multiply_frac(total_liqs, Ratio::new(my_mtsqrt, total_mtsqrt)).into();
        log::debug!(
            "added {} total liquidity out of {}!",
            deposit.outputs[0].value,
            total_liqs
        );
        state.coins.insert_coin(
            original_tx.output_coinid(0),
            CoinDataHeight {
                coin_data: deposit.outputs[0].clone(),
                height: state.height,
            },
            state.tip_906(),
        );
        if (state.network == NetID::Mainnet || state.network == NetID::Testnet)
            && state.height.0 < 978392
        {
            log::warn!("APPLYING OLD RULES THAT LEAD TO INFLATION BUG!!!!!");
            state
                .coins
                .remove_coin(deposit.output_coinid(1), state.tip_906());
        } else {
            state
                .coins
                .remove_coin(original_tx.output_coinid(1), state.tip_906());
        }
    });
}

/// Extract the deposit requests from the given `State`.
fn get_deposit_transactions<C: ContentAddrStore>(state: &UnsealedState<C>) -> Vec<Transaction> {
    state
        .transactions
        .iter()
        .cloned()
        .filter_map(|tx| {
            (tx.kind == TxKind::LiqDeposit
                && tx.outputs.len() >= 2
                && state.coins.get_coin(tx.output_coinid(0)).is_some()
                && state.coins.get_coin(tx.output_coinid(1)).is_some())
            .then_some(())?;
            let pool_key = PoolKey::from_bytes(&tx.data)?;
            (tx.outputs[0].denom == pool_key.left() && tx.outputs[1].denom == pool_key.right())
                .then_some(tx)
        })
        .collect::<Vec<_>>()
}

/// Process deposits.
fn process_deposits<C: ContentAddrStore>(mut state: UnsealedState<C>) -> UnsealedState<C> {
    let mut deposit_reqs = get_deposit_transactions(&state);
    log::trace!("{} deposit reqs", deposit_reqs.len());
    // find the pools mentioned
    let pools = extract_pool_keys_sorted(&mut deposit_reqs);
    pools.iter().for_each(|pool| {
        let mut relevant_txx: Vec<Transaction> = transactions_for_pool(&deposit_reqs, pool);
        process_deposits_for_single_pool(pool, &mut state, &mut relevant_txx);
    });

    state
}

fn process_withdrawals_for_single_pool<C: ContentAddrStore>(
    pool: &PoolKey,
    state: &mut UnsealedState<C>,
    relevant_txx: &mut [Transaction],
) {
    // sum up total liqs
    let total_liqs = relevant_txx
        .iter()
        .map(|tx| tx.outputs[0].value.0)
        .fold(0u128, |a, b| a.saturating_add(b));
    // get the state
    let mut pool_state = state.pools.get(pool).unwrap();
    let (total_left, total_write) = pool_state.withdraw(total_liqs);
    state.pools.insert(*pool, pool_state);
    // divvy up the lefts and rights
    relevant_txx.iter_mut().for_each(|deposit| {
        let coinid_0 = deposit.output_coinid(0);
        let coinid_1 = deposit.output_coinid(1);

        let my_liqs = deposit.outputs[0].value.0;
        deposit.outputs[0].denom = pool.left();
        deposit.outputs[0].value =
            multiply_frac(total_left, Ratio::new(my_liqs, total_liqs)).into();
        let synth = CoinData {
            denom: pool.right(),
            value: multiply_frac(total_write, Ratio::new(my_liqs, total_liqs)).into(),
            covhash: deposit.outputs[0].covhash,
            additional_data: deposit.outputs[0].additional_data.clone(),
        };

        state.coins.insert_coin(
            coinid_0,
            CoinDataHeight {
                coin_data: deposit.outputs[0].clone(),
                height: state.height,
            },
            state.tip_906(),
        );
        state.coins.insert_coin(
            coinid_1,
            CoinDataHeight {
                coin_data: synth,
                height: state.height,
            },
            state.tip_906(),
        );
    });
}

/// Extract the withdrawl requests from the given `State`.
fn get_withdrawal_transactions<C: ContentAddrStore>(state: &UnsealedState<C>) -> Vec<Transaction> {
    state
        .transactions
        .iter()
        .cloned()
        .filter_map(|tx| {
            (tx.kind == TxKind::LiqWithdraw
                && tx.outputs.len() == 1
                && state.coins.get_coin(tx.output_coinid(0)).is_some())
            .then_some(())?;
            let pool_key = PoolKey::from_bytes(&tx.data)?;
            state.pools.get(&pool_key)?;
            (tx.outputs[0].denom == pool_key.liq_token_denom()).then_some(tx)
        })
        .collect::<Vec<_>>()
}

/// Process deposits.
fn process_withdrawals<C: ContentAddrStore>(mut state: UnsealedState<C>) -> UnsealedState<C> {
    // find the withdrawal requests
    let mut withdraw_reqs: Vec<Transaction> = get_withdrawal_transactions(&state);
    // find the pools mentioned
    let pools = extract_pool_keys_sorted(&mut withdraw_reqs);
    pools.iter().for_each(|pool| {
        let mut relevant_txx: Vec<Transaction> = transactions_for_pool(&withdraw_reqs, pool);
        process_withdrawals_for_single_pool(pool, &mut state, &mut relevant_txx);
    });

    state
}

/// Process pegging.
fn process_pegging<C: ContentAddrStore>(mut state: UnsealedState<C>) -> UnsealedState<C> {
    // first calculate the implied sym/Erg exchange rate
    let x_sd = if state.tip_902() {
        state
            .pools
            .get(&PoolKey::new(Denom::Sym, Denom::Erg))
            .unwrap()
            .implied_price() // doscs per sym
            .recip() // syms per dosc
    } else {
        let x_s = state
            .pools
            .get(&PoolKey::new(Denom::Mel, Denom::Sym))
            .unwrap()
            .implied_price()
            .recip();
        let x_d = state
            .pools
            .get(&PoolKey::new(Denom::Mel, Denom::Erg))
            .unwrap()
            .implied_price()
            .recip();
        x_s / x_d
    };

    let throttler = if state.tip_902() { 200 } else { 1000 };

    // get the right pool
    let mut sm_pool = state
        .pools
        .get(&PoolKey::new(Denom::Mel, Denom::Sym))
        .unwrap();
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
    state
        .pools
        .insert(PoolKey::new(Denom::Mel, Denom::Sym), sm_pool);
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

    use melstructs::CoinID;
    use melvm::Covenant;

    use crate::testing::functions::{genesis_mel_coin_id, genesis_state};

    use super::*;

    #[test]
    fn math() {
        assert_eq!(multiply_frac(1000, Ratio::new(2, 1)), 2000)
    }

    #[test]
    // test a simple deposit flow
    fn simple_deposit() {
        let (my_pk, my_sk) = tmelcrypt::ed25519_keygen();
        let my_covhash = Covenant::std_ed25519_pk_legacy(my_pk).hash();
        let mut start_state = genesis_state(
            CoinID::zero_zero(),
            CoinDataHeight {
                coin_data: CoinData {
                    value: 12000.into(),
                    denom: Denom::Mel,
                    covhash: my_covhash,
                    additional_data: vec![].into(),
                },
                height: 100.into(),
            },
            Default::default(),
        );
        start_state.fee_multiplier = 1;
        // test sealing
        let mut second_state = start_state.seal(None).next_unsealed();
        // deposit the genesis as a custom-token pool
        let newcoin_tx = Transaction {
            kind: TxKind::Normal,
            inputs: vec![genesis_mel_coin_id()],
            outputs: vec![
                CoinData {
                    covhash: my_covhash,
                    value: 10000.into(),
                    denom: Denom::Mel,
                    additional_data: vec![].into(),
                },
                CoinData {
                    covhash: my_covhash,
                    value: 10000.into(),
                    denom: Denom::NewCustom,
                    additional_data: vec![].into(),
                },
            ],
            fee: 2000.into(),
            covenants: vec![Covenant::std_ed25519_pk_legacy(my_pk).to_bytes()],
            data: vec![].into(),
            sigs: vec![],
        }
        .signed_ed25519(my_sk);
        second_state.apply_tx(&newcoin_tx).unwrap();
        let pool_key = PoolKey::new(Denom::Mel, Denom::Custom(newcoin_tx.hash_nosigs()));
        let deposit_tx = Transaction {
            kind: TxKind::LiqDeposit,
            inputs: vec![newcoin_tx.output_coinid(0), newcoin_tx.output_coinid(1)],
            outputs: vec![
                CoinData {
                    covhash: my_covhash,
                    value: if pool_key.left() == Denom::Mel {
                        8000.into()
                    } else {
                        10000.into()
                    },
                    denom: pool_key.left(),
                    additional_data: vec![].into(),
                },
                CoinData {
                    covhash: my_covhash,
                    value: if pool_key.left() == Denom::Mel {
                        10000.into()
                    } else {
                        8000.into()
                    },
                    denom: pool_key.right(),
                    additional_data: vec![].into(),
                },
            ],
            fee: 2000.into(),
            covenants: vec![Covenant::std_ed25519_pk_legacy(my_pk).to_bytes()],
            data: pool_key.to_bytes(), // this is important, since it "points" to the pool
            sigs: vec![],
        }
        .signed_ed25519(my_sk);
        second_state.apply_tx(&deposit_tx).unwrap();
    }
}

use crate::SmtMapping;

/// A mapping from pool keys to internal pool states.
pub type PoolMapping<C> = SmtMapping<C, PoolKey, PoolState>;
