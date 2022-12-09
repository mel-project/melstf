use bytes::Bytes;
#[cfg(fuzzing)]
use honggfuzz::fuzz;
use melvm::Covenant;

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

#[allow(dead_code)]
fn test_once(data: &[u8]) {
    let (first, second) = data.split_at(data.len() / 2);
    let covenant = Covenant::from_bytes(first);
    if let Ok(covenant) = covenant {
        if covenant.weight() > 10000000 {
            return;
        }

        eprintln!("{:?}", covenant.to_ops());
        eprintln!("{:?}", second.len());
        covenant
            .debug_execute(&[Bytes::from(second.iter().map(|_f| 0u8).collect::<Vec<_>>()).into()]);
        assert_eq!(Covenant::from_ops(&covenant.to_ops()), covenant);
    }
}

#[cfg(not(fuzzing))]
fn main() {}
