use smallvec::SmallVec;

const CHUNK_SIZE: usize = 16;

type Chunk = SmallVec<[u8; CHUNK_SIZE]>;

/// Optimized persistent-bytes datastructure
pub struct PBytes {}
