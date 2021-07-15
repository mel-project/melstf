#![allow(clippy::float_cmp)]

use crate::{SmtMapping, TxHash};
use serde::{Deserialize, Serialize};
use tmelcrypt::Ed25519PK;

/// StakeDoc is a stake document. It encapsulates all the information needed to verify consensus proofs.
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct StakeDoc {
    /// Public key.
    pub pubkey: Ed25519PK,
    /// Starting epoch.
    pub e_start: u64,
    /// Ending epoch. This is the epoch *after* the last epoch in which the syms are effective.
    pub e_post_end: u64,
    /// Number of syms staked.
    pub syms_staked: u128,
}

/// A stake mapping
pub type StakeMapping = SmtMapping<TxHash, StakeDoc>;

impl StakeMapping {
    /// Gets the voting power, as a floating-point number, for a given public key and a given epoch.
    pub fn vote_power(&self, epoch: u64, pubkey: Ed25519PK) -> f64 {
        let mut total_votes: f64 = 1e-50;
        let mut target_votes: f64 = 0.0;

        self.val_iter().for_each(|sdoc| {
            if epoch >= sdoc.e_start && epoch < sdoc.e_post_end {
                total_votes += sdoc.syms_staked as f64;

                if sdoc.pubkey == pubkey {
                    target_votes += sdoc.syms_staked as f64;
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
    use super::*;
    use crate::{melvm, CoinData, CoinDataHeight, CoinID};
    use crate::{Denom, State};

    use std::collections::HashMap;

    use novasmt::Forest;
    use rstest::rstest;
    use tmelcrypt::Ed25519SK;

    /// Create a state using a mapping from sk to syms staked for an epoch
    fn create_state(stakers: &HashMap<Ed25519SK, u128>, epoch_start: u64) -> State {
        // Create emtpy state
        let db: Forest = Forest::new(novasmt::InMemoryBackend::default());
        let mut state: State = State::new_empty_testnet(db);

        // Insert a mel coin into state so we can transact
        let start_micromels: u128 = 10000;
        let start_conshash = melvm::Covenant::always_true().hash();

        state.coins.insert(
            CoinID {
                txhash: tmelcrypt::HashVal([0; 32]).into(),
                index: 0,
            },
            CoinDataHeight {
                coin_data: CoinData {
                    covhash: start_conshash,
                    value: start_micromels,
                    denom: Denom::Mel,
                    additional_data: Vec::new(),
                },
                height: 0,
            },
        );

        // Insert data need for staking proofs

        stakers.iter().enumerate().for_each(|(index, (sk, syms_staked))| {
            state.stakes.insert(
                tmelcrypt::hash_single(&(index as u128).to_be_bytes()).into(),
                StakeDoc {
                    pubkey: sk.to_public(),
                    e_start: epoch_start,
                    e_post_end: 1000000000,
                    syms_staked: *syms_staked,
                },
            );
        });

        state
    }

    #[rstest(
        staked_syms => [vec![100u128], vec![100u128, 10], vec![1u128, 2u128, 3u128]]
    )]
    fn test_non_staker_has_no_vote_power(staked_syms: Vec<u128>) {
        // Generate genesis block for stakers
        // let staked_syms =vec![100 as u64; 3];
        let stakers = staked_syms
            .into_iter()
            .map(|e| (tmelcrypt::ed25519_keygen().1, e))
            .collect();
        let genesis = create_state(&stakers, 0);

        // call vote_power for a key pair who is not a staker
        let (pk, _sk) = tmelcrypt::ed25519_keygen();
        let vote_power = genesis.stakes.vote_power(0, pk);

        // assert they have no vote power
        assert_eq!(vote_power, 0.0)
    }

    #[rstest(
        staked_syms => [vec![100_u128, 200_u128, 300_u128], vec![100_u128, 10], vec![1_128, 2_u128, 30_u128]]
    )]
    fn test_staker_has_correct_vote_power_in_epoch(staked_syms: Vec<u128>) {
        // Generate state for stakers
        let total_staked_syms: u128 = staked_syms.iter().sum();
        let stakers = staked_syms
            .into_iter()
            .map(|e| (tmelcrypt::ed25519_keygen().1, e))
            .collect();
        let state = create_state(&stakers, 0);

        // Check the vote power of each staker in epoch 0 has expected value
        for (sk, vote) in stakers.iter() {
            let vote_power = state.stakes.vote_power(0, sk.to_public());
            let expected_vote_power = (*vote as f64) / (total_staked_syms as f64);
            assert_eq!(expected_vote_power - vote_power, 0.0);
        }
    }

    #[rstest(
        epoch_start => [1, 2, 100]
    )]
    fn test_staker_has_no_vote_power_in_previous_epoch(epoch_start: u64) {
        // Generate state for stakers
        let staked_syms = vec![100u128; 3];
        let stakers = staked_syms
            .into_iter()
            .map(|e| (tmelcrypt::ed25519_keygen().1, e))
            .collect();
        let state = create_state(&stakers, epoch_start);

        // Check the vote power of each staker in epoch has expected value
        for (sk, _vote) in stakers.iter() {
            // Go through all previous epochs before epoch_start
            // and ensure no vote power
            for epoch in 0..epoch_start {
                let vote_power = state.stakes.vote_power(epoch, sk.to_public());
                let expected_vote_power = 0.0;
                assert_eq!(vote_power, expected_vote_power);
            }
            // Confirm vote power is non zero if at epoch_start
            let vote_power = state.stakes.vote_power(epoch_start, sk.to_public());
            let expected_vote_power = 0.0;
            assert_ne!(vote_power, expected_vote_power);
        }
    }

    #[rstest(
        staked_sym => [1, 2, 123]
    )]
    fn test_vote_power_single_staker_is_total(staked_sym: u128) {
        // Add in a single staker to get a state at epoch 0
        let (pk, sk) = tmelcrypt::ed25519_keygen();
        let mut stakers = HashMap::new();
        stakers.insert(sk, staked_sym);
        let state = create_state(&stakers, 0);

        // Ensure staker has 1.0 voting power as expected
        let expected_voting_power = 1.0;
        assert_eq!(state.stakes.vote_power(0, pk), expected_voting_power);
    }

    #[rstest(
        epoch => [0, 1, 100]
    )]
    fn test_vote_power_is_zero_no_stakers(epoch: u64) {
        let stakers = HashMap::new();
        let state = create_state(&stakers, epoch);

        let voting_power = state
            .stakes
            .vote_power(epoch, tmelcrypt::ed25519_keygen().0);
        assert_eq!(voting_power, 0.0);
    }

    #[rstest(
        staked_syms => [vec![0], vec![0; 3], vec![0; 100]]
    )]
    fn test_vote_power_is_zero_when_stakers_are_staking_zero(staked_syms: Vec<u128>) {
        // Generate state for stakers
        let stakers = staked_syms
            .into_iter()
            .map(|e| (tmelcrypt::ed25519_keygen().1, e))
            .collect();
        let state = create_state(&stakers, 0);

        // Check the vote power of each staker in epoch 0 has expected value
        for (sk, _vote) in stakers.iter() {
            let vote_power = state.stakes.vote_power(0, sk.to_public());
            assert_eq!(vote_power, 0.0);
        }
    }

    #[test]
    fn test_remove_stale_all_stale() {
        let staked_syms: Vec<u128> = vec![0; 100];

        // Generate state for stakers
        let stakers = staked_syms
            .into_iter()
            .map(|e| (tmelcrypt::ed25519_keygen().1, e))
            .collect();
        let mut state = create_state(&stakers, 0);

        // All stakes should be stale past this epoch
        state.stakes.remove_stale(100000000000);

        for (_key, value) in state.stakes.mapping.iter() {
            assert_eq!(value.as_ref(), b"");
        }
    }

    #[test]
    fn test_remove_stale_no_stale() {
        let staked_syms: Vec<u128> = vec![0; 100];

        // Generate state for stakers
        let stakers = staked_syms
            .into_iter()
            .map(|e| (tmelcrypt::ed25519_keygen().1, e))
            .collect();
        let mut state = create_state(&stakers, 0);

        // No stakes should be stale past this epoch
        state.stakes.remove_stale(100);

        for (_key, value) in state.stakes.mapping.iter() {
            assert_ne!(value.as_ref(), b"");
        }
    }
}