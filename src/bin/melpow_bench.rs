use std::time::Instant;

use themelio_stf::melpow;

fn main() {
    for difficulty in 1..25 {
        test_difficulty(difficulty)
    }
}

fn test_difficulty(difficulty: usize) {
    let start = Instant::now();
    melpow::Proof::generate(b"hello world", difficulty);
    let elapsed = start.elapsed();
    let speed = 2.0f64.powf(difficulty as f64) / elapsed.as_secs_f64();
    eprintln!(
        "difficulty {} took time {:?}, {:.2} kH/s",
        difficulty,
        elapsed,
        speed / 1000.0
    );
}
