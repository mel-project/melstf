use themelio_structs::PoolState;

use crate::{PoolKey, SmtMapping};

/// A pool
pub type PoolMapping<C> = SmtMapping<C, PoolKey, PoolState>;
