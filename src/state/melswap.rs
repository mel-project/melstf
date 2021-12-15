use crate::{PoolKey, SmtMapping, MICRO_CONVERTER};

use std::convert::TryInto;

use num::{rational::Ratio, BigInt, BigRational, BigUint};
use serde::{Deserialize, Serialize};

/// A pool
pub type PoolMapping = SmtMapping<PoolKey, PoolState>;

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct PoolState {
    pub lefts: u128,
    pub rights: u128,
    price_accum: u128,
    liqs: u128,
}

impl PoolState {
    /// Creates a new empty pool.
    pub fn new_empty() -> Self {
        Self {
            lefts: 0,
            rights: 0,
            price_accum: 0,
            liqs: 0,
        }
    }

    /// Executes a swap.
    #[must_use]
    pub fn swap_many(&mut self, lefts: u128, rights: u128) -> (u128, u128) {
        // deposit the tokens. intentionally saturate so that "overflowing" tokens are drained.
        self.lefts = self.lefts.saturating_add(lefts);
        self.rights = self.rights.saturating_add(rights);
        // "indiscriminately" use this new price to calculate how much of the other token to withdraw.
        let exchange_rate = Ratio::new(BigInt::from(self.lefts), BigInt::from(self.rights));
        let rights_to_withdraw: u128 = (BigRational::from(BigInt::from(lefts))
            / exchange_rate.clone()
            * BigRational::from(BigInt::from(995))
            / BigRational::from(BigInt::from(1000)))
        .floor()
        .numer()
        .try_into()
        .unwrap_or(u128::MAX);
        let lefts_to_withdraw: u128 = (BigRational::from(BigInt::from(rights))
            * exchange_rate
            * BigRational::from(BigInt::from(995))
            / BigRational::from(BigInt::from(1000)))
        .floor()
        .numer()
        .try_into()
        .unwrap_or(u128::MAX);
        // do the withdrawal
        self.lefts -= lefts_to_withdraw;
        self.rights -= rights_to_withdraw;

        self.price_accum = self
            .price_accum
            .overflowing_add((self.lefts).saturating_mul(MICRO_CONVERTER) / (self.rights))
            .0;

        (lefts_to_withdraw, rights_to_withdraw)
    }

    /// Deposits a set amount into the state, returning how many liquidity tokens were created.
    #[must_use]
    pub fn deposit(&mut self, lefts: u128, rights: u128) -> u128 {
        if self.liqs == 0 {
            self.lefts = lefts;
            self.rights = rights;
            self.liqs = lefts;
            lefts
        } else {
            // we first truncate mels and tokens because they can't overflow the state
            let mels = lefts.saturating_add(self.lefts) - self.lefts;
            let tokens = rights.saturating_add(self.rights) - self.rights;

            let delta_l_squared = (BigRational::from(BigInt::from(self.liqs).pow(2))
                * Ratio::new(
                    BigInt::from(mels) * BigInt::from(tokens),
                    BigInt::from(self.lefts) * BigInt::from(self.rights),
                ))
            .floor()
            .numer()
            .clone();
            let delta_l = delta_l_squared.sqrt();
            let delta_l = delta_l
                .to_biguint()
                .expect("deltaL can't possibly be negative");
            // we first convert deltaL to a u128, saturating on overflow
            let delta_l: u128 = delta_l.try_into().unwrap_or(u128::MAX);
            self.liqs = self.liqs.saturating_add(delta_l);
            self.lefts += mels;
            self.rights += tokens;
            // now we return
            delta_l
        }
    }

    /// Redeems a set amount of liquidity tokens, returning lefts and rights.
    #[must_use]
    pub fn withdraw(&mut self, liqs: u128) -> (u128, u128) {
        assert!(self.liqs >= liqs);
        let withdrawn_fraction = Ratio::new(BigUint::from(liqs), BigUint::from(self.liqs));
        let lefts =
            Ratio::new(BigUint::from(self.lefts), BigUint::from(1u32)) * withdrawn_fraction.clone();
        let rights =
            Ratio::new(BigUint::from(self.rights), BigUint::from(1u32)) * withdrawn_fraction;
        self.liqs -= liqs;
        if self.liqs == 0 {
            let toret = (self.lefts, self.rights);
            self.lefts = 0;
            self.rights = 0;
            toret
        } else {
            let toret = (
                lefts.floor().numer().try_into().unwrap(),
                rights.floor().numer().try_into().unwrap(),
            );
            self.lefts -= toret.0;
            self.rights -= toret.1;
            toret
        }
    }

    /// Returns the implied price as lefts per right.
    #[must_use]
    pub fn implied_price(&self) -> BigRational {
        Ratio::new(BigInt::from(self.lefts), BigInt::from(self.rights))
    }
    /// Returns the liquidity constant of the system.
    #[must_use]
    pub fn liq_constant(&self) -> u128 {
        self.lefts.saturating_mul(self.rights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn general() {
        let mut pool = PoolState::new_empty();
        let _ = pool.deposit(634684496, 1579230128);

        let range = 1..5;

        range.into_iter().for_each(|_index| {
            let out = pool.swap_many(100, 0);
            dbg!(pool);
            dbg!(pool.liq_constant());
            dbg!(out);
        });
    }
}
