use catvec::CatVec;
use ethnum::U256;
use themelio_structs::{CoinData, CoinDataHeight, CoinID, Denom, Header, Transaction};
use tmelcrypt::HashVal;

use super::Covenant;

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum Value {
    Int(U256),
    Bytes(CatVec<u8, 256>),
    Vector(CatVec<Value, 32>),
}

impl Value {
    pub fn into_bool(self) -> bool {
        match self {
            Value::Int(v) => v != U256::from(0u32),
            _ => true,
        }
    }

    pub fn into_int(self) -> Option<U256> {
        match self {
            Value::Int(bi) => Some(bi),
            _ => None,
        }
    }

    pub fn into_u16(self) -> Option<u16> {
        let num = self.into_int()?;
        if num > U256::from(65535u32) {
            None
        } else {
            Some(*num.low() as u16)
        }
    }

    pub fn into_truncated_u8(self) -> Option<u8> {
        let num = self.into_int()?;
        Some(*num.low() as u8)
    }
    pub fn from_bytes(bts: &[u8]) -> Self {
        Value::Bytes(bts.into())
    }
    pub fn from_bool(b: bool) -> Self {
        if b {
            Value::Int(1u32.into())
        } else {
            Value::Int(0u32.into())
        }
    }

    pub fn into_bytes(self) -> Option<CatVec<u8, 256>> {
        match self {
            Value::Bytes(bts) => Some(bts),
            _ => None,
        }
    }

    pub fn into_vector(self) -> Option<CatVec<Value, 32>> {
        match self {
            Value::Vector(vec) => Some(vec),
            _ => None,
        }
    }
}

impl From<u128> for Value {
    fn from(n: u128) -> Self {
        Value::Int(U256::from(n))
    }
}

impl From<u64> for Value {
    fn from(n: u64) -> Self {
        Value::Int(U256::from(n))
    }
}

impl From<CoinData> for Value {
    fn from(cd: CoinData) -> Self {
        Value::Vector(
            vec![
                cd.covhash.0.into(),
                cd.value.0.into(),
                cd.denom.into(),
                cd.additional_data.into(),
            ]
            .into(),
        )
    }
}

impl From<Header> for Value {
    fn from(cd: Header) -> Self {
        Value::Vector(
            vec![
                (cd.network as u64).into(),
                cd.previous.into(),
                cd.height.0.into(),
                cd.history_hash.into(),
                cd.coins_hash.into(),
                cd.transactions_hash.into(),
                cd.fee_pool.0.into(),
                cd.fee_multiplier.into(),
                cd.dosc_speed.into(),
                cd.pools_hash.into(),
                cd.stakes_hash.into(),
            ]
            .into(),
        )
    }
}

impl From<CoinDataHeight> for Value {
    fn from(cd: CoinDataHeight) -> Self {
        Value::Vector(vec![cd.coin_data.into(), cd.height.0.into()].into())
    }
}

impl From<CoinID> for Value {
    fn from(c: CoinID) -> Self {
        Value::Vector(vec![c.txhash.0.into(), Value::Int(U256::from(c.index))].into())
    }
}

impl From<Covenant> for Value {
    fn from(c: Covenant) -> Self {
        Value::Bytes(c.0.into())
    }
}

impl From<[u8; 32]> for Value {
    fn from(v: [u8; 32]) -> Self {
        Value::Bytes(v.into())
    }
}

impl From<HashVal> for Value {
    fn from(v: HashVal) -> Self {
        Value::Bytes(v.into())
    }
}

impl From<Denom> for Value {
    fn from(v: Denom) -> Self {
        Value::Bytes(v.to_bytes().into())
    }
}

impl From<Vec<u8>> for Value {
    fn from(v: Vec<u8>) -> Self {
        Value::Bytes(v.into())
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::Vector(
            v.into_iter()
                .map(|x| x.into())
                .collect::<Vec<Value>>()
                .into(),
        )
    }
}

impl From<Transaction> for Value {
    fn from(tx: Transaction) -> Self {
        Value::Vector(
            vec![
                Value::Int(U256::from(tx.kind as u8)),
                tx.inputs.into(),
                tx.outputs.into(),
                tx.fee.0.into(),
                tx.covenants.into(),
                tx.data.into(),
                tx.sigs.into(),
            ]
            .into(),
        )
    }
}
