use argh::FromArgs;
use novasmt::{Database, InMemoryCas};
use std::{fs::read_dir, path::PathBuf};
use themelio_stf::GenesisConfig;
use themelio_structs::{Block, ConsensusProof};

fn main() -> anyhow::Result<()> {
    let args: Args = argh::from_env();
    let blk_files = read_dir(args.history_path)?;
    let mut block_proofs: Vec<(Block, ConsensusProof)> = blk_files
        .map(|entry_result| {
            let path: PathBuf = entry_result.unwrap().path();
            let blk_proof = match std::fs::read(&path) {
                Ok(bytes) => stdcode::deserialize(&bytes).unwrap(),
                _ => panic!(),
            };
            blk_proof
        })
        .collect();
    block_proofs.sort_by(|a, b| a.0.header.height.cmp(&b.0.header.height));

    println!("about to apply {} historical blocks", block_proofs.len());

    let db = Database::new(InMemoryCas::default());
    let mut state = genesis_config(args.override_genesis)?
        .realize(&db)
        .seal(None);

    for blk_proof in block_proofs.iter() {
        state = state.apply_block(&blk_proof.0).unwrap();
        println!("applied block {:?}", blk_proof.0.header.height);
    }
    println!(
        "replay complete -- the state is now at height: {}",
        state.header().height,
    );

    assert_eq!(state.header().height.0 as usize, block_proofs.len());
    Ok(())
}

fn genesis_config(path: PathBuf) -> anyhow::Result<GenesisConfig> {
    let genesis_yaml = std::fs::read(&path)?;
    Ok(serde_yaml::from_slice(&genesis_yaml)?)
}

#[derive(FromArgs, PartialEq, Debug)]
/// Arguments for replaying history
pub struct Args {
    #[argh(option)]
    /// path to the history directory containing stdcode-encoded `.blk` files.
    history_path: PathBuf,

    #[argh(option)]
    /// genesis config path
    override_genesis: PathBuf,
}
