# themelio-stf: themelio's core state transition function

[![](https://img.shields.io/crates/v/themelio-stf)](https://crates.io/crates/themelio-stf)
![](https://img.shields.io/crates/l/themelio-stf)

This crate contains the data structures and core algorithms that comprise Themelio's core state transition function.
Any piece of software needing to validate Themelio transactions or answer questions like
"what happens to the Themelio state if transactions A, B, and C happen" can use this minimal-dependency crate.

## The `State` type

The most important type in the crate is `State`, and the closely associated type `SealedState`. The [yellow paper](https://docs.themelio.org/specifications/yellow/) talks about them further, but in short:

- `State` represents an **mutable** Themelio world-state and it's not directly serializable. It includes _all_ the information needed to validate new transactions and blocks, such as a SMT of all outstanding coins, Melmint parameters, etc. It has methods taking `Transaction`s etc that advance the state, as well as others to produce serializable blocks, headers, etc.
- `SealedState` represents a **sealed** state. This roughly corresponds to the notion of "the blockchain state at a given height". Blocks represent transitions from one `SealedState` to another.

## Note

This crate is the **most consensus-critical part of Themelio**, and essentially defines the entire on-chain logic of the Themelio blockchain.

Versions incompatible with the latest Themelio state are thus _all yanked_.
