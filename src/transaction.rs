use crate::{
    constants::*,
    melvm::{self, Address, Covenant},
    BlockHeight, CoinValue, HexBytes,
};

use std::{
    collections::HashMap,
    convert::TryInto,
    fmt::{Display, Formatter},
    num::ParseIntError,
    str::FromStr,
};

use arbitrary::Arbitrary;
use derive_more::{Display, From, Into};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_repr::{Deserialize_repr, Serialize_repr};
use thiserror::Error;
use tmelcrypt::{Ed25519SK, HashVal};

#[derive(
    Clone,
    Copy,
    IntoPrimitive,
    TryFromPrimitive,
    Eq,
    PartialEq,
    Arbitrary,
    Debug,
    Serialize_repr,
    Deserialize_repr,
    Hash,
)]
#[repr(u8)]
/// An enumeration of all the different possible transaction kinds. Currently contains a "faucet" kind that will be (obviously) removed in production.
pub enum TxKind {
    DoscMint = 0x50,
    Faucet = 0xff,
    LiqDeposit = 0x52,
    LiqWithdraw = 0x53,
    Normal = 0x00,
    Stake = 0x10,
    Swap = 0x51,
}

impl Display for TxKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TxKind::Normal => "Normal".fmt(f),
            TxKind::Stake => "Stake".fmt(f),
            TxKind::DoscMint => "DoscMint".fmt(f),
            TxKind::Swap => "Swap".fmt(f),
            TxKind::LiqDeposit => "LiqDeposit".fmt(f),
            TxKind::LiqWithdraw => "LiqWithdraw".fmt(f),
            TxKind::Faucet => "Faucet".fmt(f),
        }
    }
}

/// A newtype representing the hash of a transaction.
#[derive(
    Copy,
    Clone,
    Debug,
    Eq,
    PartialEq,
    Hash,
    PartialOrd,
    Ord,
    From,
    Into,
    Serialize,
    Deserialize,
    Arbitrary,
    Display,
)]
#[serde(transparent)]
pub struct TxHash(pub HashVal);

/// Transaction represents an individual, serializable Themelio transaction.
#[derive(Clone, Arbitrary, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct Transaction {
    pub kind: TxKind,
    pub inputs: Vec<CoinID>,
    pub outputs: Vec<CoinData>,
    pub fee: CoinValue,
    pub scripts: Vec<melvm::Covenant>,
    #[serde(with = "stdcode::hex")]
    pub data: Vec<u8>,
    pub sigs: Vec<HexBytes>,
}

impl Transaction {
    /// An empty transaction with kind Normal, no inputs, no fees, etc.
    pub fn empty_test() -> Self {
        Transaction {
            kind: TxKind::Normal,
            inputs: Vec::new(),
            outputs: Vec::new(),
            fee: 0.into(),
            scripts: Vec::new(),
            data: Vec::new(),
            sigs: Vec::new(),
        }
    }

    /// Creates a new transaction with the given kind, no inputs, no outputs, no nothing.
    pub fn new(kind: TxKind) -> Self {
        Self {
            kind,
            inputs: vec![],
            outputs: vec![],
            fee: 0.into(),
            scripts: vec![],
            data: vec![],
            sigs: vec![],
        }
    }

    /// Replaces the kind of the transaction
    pub fn with_kind(mut self, kind: TxKind) -> Self {
        self.kind = kind;
        self
    }

    /// Replaces the inputs of the transaction
    pub fn with_inputs(mut self, inputs: Vec<CoinID>) -> Self {
        self.inputs = inputs;
        self
    }

    /// Add an input
    pub fn add_input(mut self, input: CoinID) -> Self {
        self.inputs.push(input);
        self
    }

    /// Replaces the outputs of the transaction
    pub fn with_outputs(mut self, outputs: Vec<CoinData>) -> Self {
        self.outputs = outputs;
        self
    }

    /// Add an output
    pub fn add_output(mut self, output: CoinData) -> Self {
        self.outputs.push(output);
        self
    }

    /// Replaces the fee of the transaction
    pub fn with_fee(mut self, fee: CoinValue) -> Self {
        self.fee = fee;
        self
    }

    /// Replaces the scripts of the transaction
    pub fn with_scripts(mut self, scripts: Vec<Covenant>) -> Self {
        self.scripts = scripts;
        self
    }

    /// Add a script to the transaction
    pub fn add_script(mut self, script: Covenant) -> Self {
        self.scripts.push(script);
        self
    }

    /// Replaces the scripts of the transaction
    pub fn with_data(mut self, data: Vec<u8>) -> Self {
        self.data = data;
        self
    }

    /// Replaces the scripts of the transaction
    pub fn with_sigs(mut self, sigs: Vec<Vec<u8>>) -> Self {
        self.sigs = sigs.into_iter().map(HexBytes).collect();
        self
    }

    /// Checks whether or not the transaction is well formed, respecting coin size bounds and such. **Does not** fully validate the transaction.
    pub fn is_well_formed(&self) -> bool {
        // check bounds
        let mut output: bool = true;

        self.outputs.iter().for_each(|out| {
            if out.value > MAX_COINVAL {
                output = false;
            }
        });

        if self.fee > MAX_COINVAL {
            output = false;
        }

        if self.outputs.len() > 255 || self.inputs.len() > 255 {
            output = false;
        }

        output
    }

    /// hash_nosigs returns the hash of the transaction with a zeroed-out signature field. This is what signatures are computed against.
    pub fn hash_nosigs(&self) -> TxHash {
        let mut s = self.clone();
        s.sigs = Vec::new();
        let self_bytes = stdcode::serialize(&s).unwrap();
        tmelcrypt::hash_single(&self_bytes).into()
    }

    /// sign_ed25519 consumes the transaction, appends an ed25519 signature, and returns it.
    pub fn signed_ed25519(mut self, sk: Ed25519SK) -> Self {
        self.sigs.push(sk.sign(&self.hash_nosigs().0).into());
        self
    }

    /// total_outputs returns a HashMap mapping each type of coin to its total value. Fees will be included in the Mel cointype.
    pub fn total_outputs(&self) -> HashMap<Denom, CoinValue> {
        let mut toret: HashMap<Denom, CoinValue> = HashMap::new();

        self.outputs.iter().for_each(|output| {
            let old = toret.get(&output.denom).copied().unwrap_or_default();

            toret.insert(output.denom, old + output.value);
        });

        let old = toret.get(&Denom::Mel).copied().unwrap_or_default();
        toret.insert(Denom::Mel, old + self.fee);

        toret
    }

    /// scripts_as_map returns a HashMap mapping the hash of each script in the transaction to the script itself.
    pub fn script_as_map(&self) -> HashMap<Address, Covenant> {
        self.scripts.iter().map(|script| {
            (script.hash(), script.clone())
        }).collect::<HashMap<Address, Covenant>>()
    }

    /// Returns the minimum fee of the transaction at a given fee multiplier, with a given "ballast".
    pub fn base_fee(&self, fee_multiplier: u128, ballast: u128) -> CoinValue {
        ((self.weight().saturating_add(ballast)).saturating_mul(fee_multiplier) >> 16).into()
    }

    /// Returns the weight of the transaction.
    pub fn weight(&self) -> u128 {
        let raw_length = stdcode::serialize(self).unwrap().len() as u128;
        let script_weights: u128 = self
            .scripts
            .iter()
            .map(|scr| scr.weight().unwrap_or_default())
            .sum();
        // we price in the net state "burden".
        // how much is that? let's assume that history is stored for 1 month. this means that "stored" bytes are around 240 times more expensive than "temporary" bytes.
        // we also take into account that stored stuff is probably going to be stuffed into something much cheaper (e.g. HDD rather than RAM), almost certainly more than 24 times cheaper.
        // so it's probably "safe-ish" to say that stored things are 10 times more expensive than temporary things.
        // econ efficiency/market stability wise it's probably okay to overprice storage, but probably not okay to underprice it.
        // blockchain-spamming-as-HDD arbitrage is going to be really bad for the blockchain.
        // penalize 1000 for every output and boost 1000 for every input. "non-refundable" because the fee can't be subzero
        let output_penalty = self.outputs.len() as u128 * 1000;
        let input_boon = self.inputs.len() as u128 * 1000;

        raw_length
            .saturating_add(script_weights)
            .saturating_add(output_penalty)
            .saturating_sub(input_boon)
    }

    /// Convenience function that constructs a CoinID that points to a certain output of this transaction. Panics if the index is out of bounds.
    pub fn output_coinid(&self, index: u8) -> CoinID {
        assert!((index as usize) < self.outputs.len());
        CoinID {
            txhash: self.hash_nosigs(),
            index,
        }
    }

    /// Convenience function that applies the correct fee.
    /// Call this *before* signing the transaction,
    /// with a ballast that's an upper bound on the number of bytes
    /// added to the transaction as signatures. 100 is a good value for a ballast.
    /// Provide the index of the output to deduct from;
    /// returns None if the output doesn't have enough money to cover fees.
    ///
    /// **Deprecated**: you should use TransactionBuilder instead.
    #[deprecated]
    pub fn applied_fee(
        mut self,
        fee_multiplier: u128,
        ballast: u128,
        deduct_from_idx: usize,
    ) -> Option<Self> {
        let delta_fee = self.base_fee(fee_multiplier, ballast);
        self.fee += delta_fee;
        let deduct_from = self.outputs.get_mut(deduct_from_idx)?;
        deduct_from.value = deduct_from.value.0.checked_sub(delta_fee.0)?.into();
        Some(self)
    }
}

#[derive(
    Serialize, Deserialize, Clone, Debug, Copy, Arbitrary, Ord, PartialOrd, Eq, PartialEq, Hash,
)]
/// A coin ID, consisting of a transaction hash and index. Uniquely identifies a coin in Themelio's history.
pub struct CoinID {
    pub txhash: TxHash,
    pub index: u8,
}

impl CoinID {
    /// Creates a new CoinID
    pub fn new(txhash: TxHash, index: u8) -> Self {
        CoinID { txhash, index }
    }
}

#[derive(Error, Debug, Clone)]
pub enum ParseCoinIDError {
    #[error("could not split into txhash-index")]
    CannotSplit,
    #[error("hex error ({0})")]
    HexError(#[from] hex::FromHexError),
    #[error("parse int error ({0})")]
    ParseIntError(#[from] ParseIntError),
}

impl FromStr for CoinID {
    type Err = ParseCoinIDError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let splitted = s.split('-').collect::<Vec<_>>();
        if splitted.len() != 2 {
            return Err(ParseCoinIDError::CannotSplit);
        }
        let txhash: HashVal = splitted[0].parse()?;
        let index: u8 = splitted[1].parse()?;
        Ok(CoinID {
            txhash: txhash.into(),
            index,
        })
    }
}

impl Display for CoinID {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.txhash.fmt(f)?;
        '-'.fmt(f)?;
        self.index.fmt(f)
    }
}

impl CoinID {
    /// The genesis coin of "zero-zero".
    pub fn zero_zero() -> Self {
        Self {
            txhash: tmelcrypt::HashVal::default().into(),
            index: 0,
        }
    }

    /// The pseudo-coin-ID for the proposer reward for the given height.
    pub fn proposer_reward(height: BlockHeight) -> Self {
        CoinID {
            txhash: tmelcrypt::hash_keyed(b"reward_coin_pseudoid", &height.0.to_be_bytes()).into(),
            index: 0,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Arbitrary, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
/// The data bound to a coin ID. Contains the "contents" of a coin, i.e. its constraint hash, value, and coin type.
pub struct CoinData {
    #[serde(with = "stdcode::asstr")]
    pub covhash: Address,
    pub value: CoinValue,
    // #[serde(with = "stdcode::hex")]
    pub denom: Denom,
    #[serde(with = "stdcode::hex")]
    pub additional_data: Vec<u8>,
}

impl CoinData {
    pub fn additional_data_hex(&self) -> String {
        hex::encode(&self.additional_data)
    }
}

#[derive(Clone, Arbitrary, Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Copy)]
pub enum Denom {
    Mel,
    Sym,
    NomDosc,

    NewCoin,
    Custom(TxHash),
}

impl Display for Denom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s: String = match self {
            Denom::Mel => "MEL".into(),
            Denom::Sym => "SYM".into(),
            Denom::NomDosc => "N-DOSC".into(),
            Denom::NewCoin => "(NEWCOIN)".into(),
            Denom::Custom(hash) => format!("CUSTOM-{}", hash.0),
        };
        s.fmt(f)
    }
}

#[derive(Error, Debug, Clone)]
pub enum ParseDenomError {
    #[error("Invalid denom name")]
    Invalid,
    #[error("hex error ({0})")]
    HexError(#[from] hex::FromHexError),
}

impl FromStr for Denom {
    type Err = ParseDenomError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "MEL" => Ok(Denom::Mel),
            "SYM" => Ok(Denom::Sym),
            "N-DOSC" => Ok(Denom::NomDosc),
            "(NEWCOIN)" => Ok(Denom::NewCoin),
            other => {
                let splitted = other.split('-').collect::<Vec<_>>();
                if splitted.len() != 2 || splitted[0] != "CUSTOM" {
                    return Err(ParseDenomError::Invalid);
                }
                let hv: HashVal = splitted[1].parse()?;
                Ok(Denom::Custom(TxHash(hv)))
            }
        }
    }
}

impl Denom {
    pub fn to_bytes(self) -> Vec<u8> {
        match self {
            Self::Mel => b"m".to_vec(),
            Self::Sym => b"s".to_vec(),
            Self::NomDosc => b"d".to_vec(),
            Self::NewCoin => b"".to_vec(),
            Self::Custom(hash) => hash.0.to_vec(),
        }
    }

    pub fn from_bytes(vec: &[u8]) -> Option<Self> {
        Some(match vec {
            b"m" => Self::Mel,
            b"s" => Self::Sym,
            b"d" => Self::NomDosc,

            b"" => Self::NewCoin,
            other => Self::Custom(HashVal(other.try_into().ok()?).into()),
        })
    }
}

impl Serialize for Denom {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        DenomInner(self.to_bytes()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Denom {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let inner = <DenomInner>::deserialize(deserializer)?;
        Denom::from_bytes(&inner.0)
            .ok_or_else(|| serde::de::Error::custom("not the right format for a Denom"))
    }
}

/// A coin denomination, like mel, sym, etc.
#[derive(Serialize, Deserialize, Clone, Arbitrary, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
struct DenomInner(#[serde(with = "stdcode::hex")] Vec<u8>);

#[derive(Serialize, Deserialize, Clone, Arbitrary, Debug, Eq, PartialEq, Hash)]
/// A `CoinData` but coupled with a block height. This is what actually gets stored in the global state, allowing constraints and the validity-checking algorithm to easily access the age of a coin.
pub struct CoinDataHeight {
    pub coin_data: CoinData,
    pub height: BlockHeight,
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::{melvm, CoinData, Transaction, MAX_COINVAL};
    use crate::Denom;
    // use std::sync::Arc;

    lazy_static!{
        pub static ref VALID_TRANSACTION: Vec<Transaction> = crate::testing::functions::valid_txx(tmelcrypt::ed25519_keygen());
    }

    #[test]
    fn test_is_well_formed() {
        VALID_TRANSACTION.iter().for_each(|valid_tx| {
            assert!(valid_tx.is_well_formed());
        });
    }

    #[test]
    fn test_is_not_well_formed_if_value_gt_max() {
        // Extract out first coin data from first transaction in valid transactions
        let valid_tx = VALID_TRANSACTION.get(0).unwrap().clone();
        let valid_outputs = valid_tx.outputs;
        let valid_output = valid_outputs.get(0).unwrap().clone();

        // Create an invalid tx by setting an invalid output value
        let invalid_output_value = MAX_COINVAL + 1.into();
        let invalid_output = CoinData {
            value: invalid_output_value,
            ..valid_output
        };
        let invalid_outputs = vec![invalid_output];
        let invalid_tx = Transaction {
            outputs: invalid_outputs,
            ..valid_tx
        };

        // Ensure transaction is not well formed
        assert!(!invalid_tx.is_well_formed());
    }

    #[test]
    fn test_is_not_well_formed_if_fee_gt_max() {
        let offsets: [u128; 3] = [1, 2, 100];

        offsets.iter().for_each(|offset| {
            // Extract out first coin data from first transaction in valid transactions
            let valid_tx = VALID_TRANSACTION.get(0).unwrap().clone();

            // Create an invalid tx by setting an invalid fee value
            let invalid_tx = Transaction {
                fee: MAX_COINVAL + crate::CoinValue::from(*offset),
                ..valid_tx
            };

            // Ensure transaction is not well formed
            assert!(!invalid_tx.is_well_formed());
        });
    }

    #[test]
    fn test_is_not_well_formed_if_io_gt_max() {
        let offsets: [usize; 3] = [1, 2, 100];

        offsets.iter().for_each(|offset| {
            // Extract out first coin data from first transaction in valid transactions
            let valid_tx = VALID_TRANSACTION.get(0).unwrap().clone();
            let valid_outputs = valid_tx.outputs;
            let valid_output = valid_outputs.get(0).unwrap().clone();

            // Create an invalid tx by setting an invalid output value
            let invalid_output_count = 255 + offset;
            let invalid_outputs = vec![valid_output; invalid_output_count];
            let invalid_tx = Transaction {
                outputs: invalid_outputs,
                ..valid_tx
            };

            // Ensure transaction is not well formed
            assert!(!invalid_tx.is_well_formed());
        });

        // TODO: add case for input_count exceeding limit
    }

    #[test]
    fn test_hash_no_sigs() {
        // Check that valid transaction has a non zero number of signatures
        let valid_tx = VALID_TRANSACTION.get(0).unwrap().clone();
        assert_ne!(valid_tx.sigs.len(), 0);

        // Create a transaction from it which has no signatures
        let mut no_sigs_tx = valid_tx.clone();
        no_sigs_tx.sigs = vec![];

        // Create a transaction from valid which has another signature
        let more_sig_tx = valid_tx.clone();
        let new_sk = tmelcrypt::ed25519_keygen().1;
        let more_sig_tx = more_sig_tx.signed_ed25519(new_sk);

        // Ensure they all hash to same value
        let h1 = valid_tx.hash_nosigs();
        let h2 = no_sigs_tx.hash_nosigs();
        let h3 = more_sig_tx.hash_nosigs();

        assert_eq!(h1, h2);
        assert_eq!(h1, h3);
    }

    #[test]
    fn test_sign_sigs() {
        // Create a transaction from it which has no signatures
        let valid_tx = VALID_TRANSACTION.get(0).unwrap().clone();
        assert_ne!(valid_tx.sigs.len(), 0);
        let mut no_sigs_tx = valid_tx;
        no_sigs_tx.sigs = vec![];
        assert_eq!(no_sigs_tx.sigs.len(), 0);

        // sign it N times
        let mut mult_signature_tx = no_sigs_tx;
        let n = 5;

        vec![tmelcrypt::ed25519_keygen(); n].iter().for_each(|(_pk, sk)| {
            mult_signature_tx = mult_signature_tx.clone().signed_ed25519(*sk);
        });

        // verify it has N signatures
        assert_eq!(mult_signature_tx.sigs.len(), n);

        // sign it M times
        let m = 8;
        vec![tmelcrypt::ed25519_keygen(); m].iter().for_each(|(_pk, sk)| {
            mult_signature_tx = mult_signature_tx.clone().signed_ed25519(*sk);
        });

        // verify it has N + M signatures
        assert_eq!(mult_signature_tx.sigs.len(), n + m);
    }

    #[test]
    fn test_sign_sigs_and_verify() {
        // Create a transaction from it which has no signatures
        let valid_tx = VALID_TRANSACTION.get(0).unwrap().clone();
        assert_ne!(valid_tx.sigs.len(), 0);
        let mut no_sigs_tx = valid_tx;
        no_sigs_tx.sigs = vec![];
        assert_eq!(no_sigs_tx.sigs.len(), 0);

        // create two key pairs
        let (pk1, sk1) = tmelcrypt::ed25519_keygen();
        let (pk2, sk2) = tmelcrypt::ed25519_keygen();

        // sign it
        let mut tx = no_sigs_tx;
        tx = tx.signed_ed25519(sk1);
        tx = tx.signed_ed25519(sk2);

        // verify it is signed by expected keys
        let sig1 = tx.sigs[0].clone();
        let sig2 = tx.sigs[1].clone();

        pk1.verify(&tx.hash_nosigs().0.to_vec(), &sig1);
        pk2.verify(&tx.hash_nosigs().0.to_vec(), &sig2);

        assert_eq!(tx.sigs.len(), 2);
    }

    #[test]
    fn test_total_output() {
        // create transaction
        let mut valid_tx = VALID_TRANSACTION.get(0).unwrap().clone();
        let (pk, _sk) = tmelcrypt::ed25519_keygen();
        let scr = melvm::Covenant::std_ed25519_pk_legacy(pk);

        // insert coins
        let val1 = 100;
        let val2 = 200;
        valid_tx.outputs = vec![
            CoinData {
                covhash: scr.hash(),
                value: val1.into(),
                denom: Denom::NewCoin,
                additional_data: vec![],
            },
            CoinData {
                covhash: scr.hash(),
                value: val2.into(),
                denom: Denom::NewCoin,
                additional_data: vec![],
            },
        ];

        // Check total is valid
        let value_by_coin_type = valid_tx.total_outputs();
        let total: u128 = value_by_coin_type.iter().map(|(_k, v)| v.0).sum();

        let fee = 1577000; // Temporary hack
        assert_eq!(total, val1 + val2 + fee);
    }

    #[test]
    fn test_script_as_map() {
        // create transaction
        let valid_tx = VALID_TRANSACTION.get(0).unwrap().clone();
        let (pk, _sk) = tmelcrypt::ed25519_keygen();
        let _scr = melvm::Covenant::std_ed25519_pk_legacy(pk);

        // add scripts

        // call script_as_map
        let _script_map = valid_tx.script_as_map();

        // verify num scripts = length of returned hashmap

        // verify hashes match expected value
    }

    #[test]
    fn test_weight_adjust() {
        // create a transaction

        // call weight with 0 and store

        // call weight with N as adjust and ensure difference is adjust
    }

    #[test]
    fn test_weight_does_not_exceed_max_u64() {
        // create a transaction

        // call weight with max u64 size

        // verify result is max u64 size
    }
}
