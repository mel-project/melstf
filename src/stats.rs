use std::{sync::atomic::Ordering, time::Instant};

use atomic_float::AtomicF64;

/// Tracks the total time spent applying transactions.

pub static STAT_APPLY_SECS: StatCounter = StatCounter {
    counter: AtomicF64::new(0.0),
};

/// Tracks the total time spent obtaining values from SMTs.
pub static STAT_SMT_GET_SECS: StatCounter = StatCounter {
    counter: AtomicF64::new(0.0),
};

/// Tracks the total time spent inserting values into SMTs.
pub static STAT_SMT_INSERT_SECS: StatCounter = StatCounter {
    counter: AtomicF64::new(0.0),
};

/// Tracks the total time spent validating covenants.
pub static STAT_MELVM_RUNTIME_SECS: StatCounter = StatCounter {
    counter: AtomicF64::new(0.0),
};

/// Tracks the total time spent validating melpow proofs.
pub static STAT_MELPOW_SECS: StatCounter = StatCounter {
    counter: AtomicF64::new(0.0),
};

/// Statistics counter. Tracks some floating-point metric.
pub struct StatCounter {
    counter: AtomicF64,
}

impl StatCounter {
    /// Reads the value out.
    pub fn value(&self) -> f64 {
        self.counter.load(Ordering::Relaxed)
    }

    /// Increments this stat by some number.
    pub(crate) fn incr(&self, delta: f64) {
        self.counter.fetch_add(delta, Ordering::Relaxed);
    }

    /// Create a duration-based timer.
    pub(crate) fn timer_secs(&self, name: &'static str) -> StatTimer<'_> {
        StatTimer {
            r: self,
            name,
            start: Instant::now(),
        }
    }
}

#[allow(unused)]
/// A timer that increments the statistic when dropped
pub struct StatTimer<'a> {
    r: &'a StatCounter,
    name: &'static str,
    start: Instant,
}

impl<'a> Drop for StatTimer<'a> {
    fn drop(&mut self) {
        self.r.incr(self.start.elapsed().as_secs_f64())
    }
}
