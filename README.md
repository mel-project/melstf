# themelio-stf: themelio's core state transition function

[![](https://img.shields.io/crates/v/themelio-stf)](https://crates.io/crates/themelio-stf)
![](https://img.shields.io/crates/l/themelio-stf)

This crate contains the data structures and core algorithms that comprise Themelio's core state transition function.
Any piece of software needing to validate Themelio transactions or answer questions like
"what happens to the Themelio state if transactions A, B, and C happen" can use this minimal-depedency crate.

The most important type in the c

- `State` represents a full Themelio world-state and it's not directly serializable. It includes _all_ the information needed to validate new transactions and blocks, such as a SMT of all outstanding coins, Melmint parameters, etc. It has methods taking `Transaction`s etc that advance the state, as well as others to produce serializable blocks, headers, etc.
- `Transaction` represents a serializable Themelio transaction. It has some helper methods to count coins, estimate fees, etc, largely to help build wallets.
- `StakeDoc`, which every `State` includes, encapsulates the Symphonia epoch-based stake information.
- `SmtMapping` represents a type-safe SMT-backed mapping that is extensively used within the crate.

This is the **most consensus-critical part of Themelio**, and essentially defines the entire on-chain logic of the Themelio blockchain.
