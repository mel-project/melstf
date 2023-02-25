use argh::FromArgs;
use melstf::GenesisConfig;
use melstructs::{Block, ConsensusProof};
use novasmt::{Database, InMemoryCas};
use std::{fs::read_dir, path::PathBuf};

fn main() -> anyhow::Result<()> {
    let args: Args = argh::from_env();
    let blk_files = read_dir(&args.history_path)?;
    // get the filenames first, in sorted order
    let filenames = {
        let raw_filenames: anyhow::Result<Vec<String>> = blk_files
            .map(|entry| anyhow::Ok(entry?.file_name().to_string_lossy().to_string()))
            .collect();
        let mut buf = raw_filenames?;
        buf.sort_unstable();
        buf
    };
    println!(
        "about to apply {} historical blocks from {:?}",
        filenames.len(),
        &args.history_path
    );

    let db = Database::new(InMemoryCas::default());
    let mut state = genesis_config(args.override_genesis)?
        .realize(&db)
        .seal(None);

    for blk_proof in filenames {
        let mut fpath = args.history_path.clone();
        fpath.push(blk_proof);
        let (blk, _proof): (Block, ConsensusProof) = stdcode::deserialize(&std::fs::read(&fpath)?)?;
        println!("read block {}", blk.header.height);
        state = state.apply_block(&blk)?;
        println!("applied block {}", blk.header.height);
    }
    println!(
        "replay complete -- the state is now at height: {}",
        state.header().height,
    );
    Ok(())
}

fn genesis_config(path: Option<PathBuf>) -> anyhow::Result<GenesisConfig> {
    if let Some(path) = path {
        let genesis_yaml = std::fs::read(&path)?;
        Ok(serde_yaml::from_slice(&genesis_yaml)?)
    } else {
        Ok(GenesisConfig::std_mainnet())
    }
}

#[derive(FromArgs, PartialEq, Eq, Debug)]
/// Arguments for replaying history
pub struct Args {
    #[argh(option)]
    /// path to the history directory containing stdcode-encoded `.blk` files.
    history_path: PathBuf,

    #[argh(option)]
    /// genesis config path
    override_genesis: Option<PathBuf>,
}
