#![allow(clippy::float_cmp)]

use crate::SmtMapping;
use novasmt::ContentAddrStore;
use themelio_structs::{StakeDoc, TxHash};
use tmelcrypt::Ed25519PK;

/// A stake mapping
pub type StakeMapping<C> = SmtMapping<C, TxHash, StakeDoc>;

impl<C: ContentAddrStore> StakeMapping<C> {
    /// Gets the voting power, as a floating-point number, for a given public key and a given epoch.
    pub fn vote_power(&self, epoch: u64, pubkey: Ed25519PK) -> f64 {
        let mut total_votes = 1e-50;
        let mut target_votes = 0.0;

        self.val_iter().for_each(|sdoc| {
            if epoch >= sdoc.e_start && epoch < sdoc.e_post_end {
                total_votes += sdoc.syms_staked.0 as f64;
                if sdoc.pubkey == pubkey {
                    target_votes += sdoc.syms_staked.0 as f64;
                }
            }
        });

        target_votes / total_votes
    }

    /// Filter out all the elements that no longer matter.
    pub fn remove_stale(&mut self, epoch: u64) {
        let stale_key_hashes: Vec<[u8; 32]> = self
            .mapping
            .iter()
            .filter_map(|(kh, v)| {
                let v: StakeDoc = stdcode::deserialize(&v).unwrap();
                if epoch > v.e_post_end {
                    Some(kh)
                } else {
                    None
                }
            })
            .collect::<Vec<[u8; 32]>>();

        stale_key_hashes.iter().for_each(|stale_key| {
            self.mapping.insert(*stale_key, Default::default());
        });
    }
}

#[cfg(test)]
mod tests {
    use themelio_structs::CoinValue;

    use crate::testing::functions::create_state;

    use std::collections::HashMap;

    #[test]
    fn test_non_staker_has_no_vote_power() {
        let mut staked_sym_group: Vec<Vec<u128>> = Vec::new();
        staked_sym_group.push(vec![100]);
        staked_sym_group.push(vec![100, 10]);
        staked_sym_group.push(vec![1, 2, 3]);

        staked_sym_group.iter().for_each(|staked_syms| {
            // Generate genesis block for stakers
            // let staked_syms =vec![100 as u64; 3];
            let stakers = staked_syms
                .iter()
                .map(|e| (tmelcrypt::ed25519_keygen().1, CoinValue(*e)))
                .collect();
            let genesis = create_state(&stakers, 0);

            // call vote_power for a key pair who is not a staker
            let (pk, _sk) = tmelcrypt::ed25519_keygen();
            let vote_power = genesis.stakes.vote_power(0, pk);

            // assert they have no vote power
            assert_eq!(vote_power, 0.0)
        });
    }

    #[test]
    fn test_staker_has_correct_vote_power_in_epoch() {
        let mut staked_sym_group: Vec<Vec<u128>> = Vec::new();
        staked_sym_group.push(vec![100, 200, 300]);
        staked_sym_group.push(vec![100, 10]);
        staked_sym_group.push(vec![1, 2, 30]);

        staked_sym_group.iter().for_each(|staked_syms| {
            // Generate state for stakers
            let total_staked_syms: u128 = staked_syms.iter().sum();
            let stakers = staked_syms
                .iter()
                .map(|e| (tmelcrypt::ed25519_keygen().1, CoinValue(*e)))
                .collect();
            let state = create_state(&stakers, 0);

            // Check the vote power of each staker in epoch 0 has expected value
            stakers.iter().for_each(|(sk, vote)| {
                let vote_power = state.stakes.vote_power(0, sk.to_public());
                let expected_vote_power = (vote.0 as f64) / (total_staked_syms as f64);
                assert_eq!(expected_vote_power - vote_power, 0.0);
            });
        });
    }

    #[test]
    fn test_staker_has_no_vote_power_in_previous_epoch() {
        let epoch_group: [u64; 3] = [1, 2, 100];

        epoch_group.into_iter().for_each(|epoch_start| {
            // Generate state for stakers
            let staked_syms = vec![100u128; 3];
            let stakers = staked_syms
                .into_iter()
                .map(|e| (tmelcrypt::ed25519_keygen().1, CoinValue(e)))
                .collect();
            let state = create_state(&stakers, epoch_start);

            // Check the vote power of each staker in epoch has expected value
            stakers.iter().for_each(|(sk, _vote)| {
                // Go through all previous epochs before epoch_start
                // and ensure no vote power
                let range = 0..epoch_start;

                range.into_iter().for_each(|epoch| {
                    let vote_power = state.stakes.vote_power(epoch, sk.to_public());
                    let expected_vote_power = 0.0;
                    assert_eq!(vote_power, expected_vote_power);
                });

                // Confirm vote power is non zero if at epoch_start
                let vote_power = state.stakes.vote_power(epoch_start, sk.to_public());
                let expected_vote_power = 0.0;
                assert_ne!(vote_power, expected_vote_power);
            });
        });
    }

    #[test]
    fn test_vote_power_single_staker_is_total() {
        let staked_sym_group: [u128; 3] = [1, 2, 123];

        staked_sym_group.iter().for_each(|staked_sym| {
            // Add in a single staker to get a state at epoch 0
            let (pk, sk) = tmelcrypt::ed25519_keygen();
            let mut stakers = HashMap::new();
            stakers.insert(sk, CoinValue(*staked_sym));
            let state = create_state(&stakers, 0);

            // Ensure staker has 1.0 voting power as expected
            let expected_voting_power = 1.0;
            assert_eq!(state.stakes.vote_power(0, pk), expected_voting_power);
        });
    }

    #[test]
    fn test_vote_power_is_zero_no_stakers() {
        let epoch_group: [u64; 3] = [0, 1, 100];

        epoch_group.into_iter().for_each(|epoch| {
            let stakers = HashMap::new();
            let state = create_state(&stakers, epoch);

            let voting_power = state
                .stakes
                .vote_power(epoch, tmelcrypt::ed25519_keygen().0);
            assert_eq!(voting_power, 0.0);
        });
    }

    #[test]
    fn test_vote_power_is_zero_when_stakers_are_staking_zero() {
        let mut staked_sym_group: Vec<Vec<u128>> = Vec::new();
        staked_sym_group.push(vec![0]);
        staked_sym_group.push(vec![0; 3]);
        staked_sym_group.push(vec![0; 100]);

        staked_sym_group.iter().for_each(|staked_syms| {
            // Generate state for stakers
            let stakers = staked_syms
                .iter()
                .map(|e| (tmelcrypt::ed25519_keygen().1, CoinValue(*e)))
                .collect();
            let state = create_state(&stakers, 0);

            // Check the vote power of each staker in epoch 0 has expected value
            stakers.iter().for_each(|(sk, _vote)| {
                let vote_power = state.stakes.vote_power(0, sk.to_public());
                assert_eq!(vote_power, 0.0);
            });
        });
    }

    #[test]
    fn test_remove_stale_all_stale() {
        let staked_syms: Vec<u128> = vec![0; 100];

        // Generate state for stakers
        let stakers = staked_syms
            .into_iter()
            .map(|e| (tmelcrypt::ed25519_keygen().1, CoinValue(e)))
            .collect();
        let mut state = create_state(&stakers, 0);

        // All stakes should be stale past this epoch
        state.stakes.remove_stale(100000000000);

        state.stakes.mapping.iter().for_each(|(_key, value)| {
            assert_eq!(value.as_ref(), b"");
        });
    }

    #[test]
    fn test_remove_stale_no_stale() {
        let staked_syms: Vec<u128> = vec![0; 100];

        // Generate state for stakers
        let stakers = staked_syms
            .into_iter()
            .map(|e| (tmelcrypt::ed25519_keygen().1, CoinValue(e)))
            .collect();
        let mut state = create_state(&stakers, 0);

        // No stakes should be stale past this epoch
        state.stakes.remove_stale(100);

        state.stakes.mapping.iter().for_each(|(_key, value)| {
            assert_ne!(value.as_ref(), b"");
        });
    }
}
