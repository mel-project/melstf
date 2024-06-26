# melstructs: Mel's core state transition function

[![](https://img.shields.io/crates/v/melstructs)](https://crates.io/crates/melstructs)
![](https://img.shields.io/crates/l/melstructs)

This crate contains the data structures and core algorithms that comprise Mel's core state transition function.
Any piece of software needing to validate Mel transactions or answer questions like
"what happens to the Mel state if transactions A, B, and C happen" can use this minimal-dependency crate.

## The `State` type

The most important type in the crate is `State`, and the closely associated type `SealedState`. The [yellow paper](https://docs.themelio.org/specifications/yellow/) talks about them further, but in short:

- `State` represents an **mutable** Mel world-state and it's not directly serializable. It includes _all_ the information needed to validate new transactions and blocks, such as a SMT of all outstanding coins, Melmint parameters, etc. It has methods taking `Transaction`s etc that advance the state, as well as others to produce serializable blocks, headers, etc.
- `SealedState` represents a **sealed** state. This roughly corresponds to the notion of "the blockchain state at a given height". Blocks represent transitions from one `SealedState` to another.

## Note

This crate is the **most consensus-critical part of Mel**, and essentially defines the entire on-chain logic of the Mel blockchain.

Versions incompatible with the latest Mel state are thus _all yanked_.

## Example Usage in `melnode`

To illustrate STF's usage, let's look at [`melnode`](https://github.com/mel-project/melnode) as an example.

When a node is run for the first time, it will instantiate storage to keep track of the `State`. It does this by reading it from persisted history data, or create a new genesis state.

During it's lifetime, the node will need to update it's current knowledge of the `State` by doing things like handling incoming `apply_tx` RPCs, or syncing up its blocks (`apply_block`) with other nodes.

For example, an incoming `apply_tx` RPC call to the node will take the current provisional `State` and call `State::apply_tx`. If the node is a staker, this provisional state is later used to participate in the consensus algorithm (e.g. `streamlette`).

When a slower node syncs its state with a peer at a higher height, it will continuously call `apply_block` on its current highest state.
