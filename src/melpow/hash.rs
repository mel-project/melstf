use crate::melpow::node::SVec;

pub fn bts_key(bts: &[u8], key: &[u8]) -> SVec<u8> {
    SVec::from_slice(&tmelcrypt::hash_keyed(key, bts))
}

#[derive(Default)]
pub struct Accumulator {
    b3h: blake3::Hasher,
}

impl Accumulator {
    pub fn new(key: &[u8]) -> Self {
        Accumulator {
            b3h: blake3::Hasher::new_keyed(blake3::hash(key).as_bytes()),
        }
    }

    #[inline]
    pub fn add(&mut self, bts: &[u8]) -> &mut Self {
        let blen = (bts.len() as u64).to_be_bytes();
        self.b3h.update(&blen);
        self.b3h.update(bts);
        self
    }

    #[inline]
    pub fn hash(&self) -> SVec<u8> {
        SVec::from_slice(self.b3h.finalize().as_bytes())
    }
}
