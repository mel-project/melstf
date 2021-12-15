#[cfg(fuzzing)]
use honggfuzz::fuzz;
use themelio_stf::melvm::Covenant;

#[cfg(fuzzing)]
fn main() {
    use env_logger::Env;
    env_logger::Builder::from_env(Env::default().default_filter_or("blkstructs")).init();
    // Here you can parse `std::env::args and
    // setup / initialize your project

    // You have full control over the loop but
    // you're supposed to call `fuzz` ad vitam aeternam
    loop {
        // The fuzz macro gives an arbitrary object (see `arbitrary crate`)
        // to a closure-like block of code.
        // For performance reasons, it is recommended that you use the native type
        // `&[u8]` when possible.
        // Here, this slice will contain a "random" quantity of "random" data.
        fuzz!(|data: &[u8]| { test_once(data) });
    }
}

fn test_once(data: &[u8]) {
    let (first, second) = data.split_at(data.len() / 2);
    let covenant = Covenant(first.to_vec());
    if let Ok(weight) = covenant.weight() {
        if weight > 10000000 {
            return;
        }
    }
    eprintln!("{:?}", covenant.to_ops());
    eprintln!("{:?}", second.len());
    covenant.debug_run_without_transaction(&[second.iter().map(|_f| 0u8).collect::<Vec<_>>().into()]);
    if let Ok(ops) = covenant.to_ops() {
        assert_eq!(Covenant::from_ops(&ops).unwrap(), covenant);
    }
}

#[cfg(not(fuzzing))]
fn main() {}
