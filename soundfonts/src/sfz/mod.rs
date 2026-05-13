use std::path::PathBuf;

use self::{parse::parse_tokens_resolved, region::parse_sf_root};

mod grammar;
mod parse;
mod region;
pub use parse::{SfzParseError, SfzValidationError};
pub use region::{AmpegEnvelopeParams, RegionParams, TriggerType};

/// Parses an SFZ file and returns its regions in a vector.
pub fn parse_soundfont(sfz_path: impl Into<PathBuf>) -> Result<Vec<RegionParams>, SfzParseError> {
    let sfz_path = sfz_path.into();
    let sfz_path: PathBuf = sfz_path
        .canonicalize()
        .map_err(|_| SfzParseError::FailedToReadFile(sfz_path))?;

    let tokens = parse_tokens_resolved(&sfz_path)?;

    // Unwrap here is safe because the path is confirmed to be a file due to `parse_all_tokens`
    // and therefore it will always have a parent folder. The path is also canonicalized.
    let parent_path = sfz_path.parent().unwrap().into();

    let regions = parse_sf_root(tokens.into_iter(), parent_path);

    Ok(regions)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::parse_soundfont;

    fn create_temp_dir(label: &str) -> PathBuf {
        let unique = format!(
            "xsynth-sfz-test-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn parse_soundfont_resolves_includes_and_defines() {
        let dir = create_temp_dir("include-define");
        let sfz_path = dir.join("instrument.sfz");
        let inc_path = dir.join("include.sfz");
        let sample_path = dir.join("samples").join("snare.wav");

        write(&sample_path, "");
        write(
            &inc_path,
            r#"
<group>
lovel=1
sample=$SAMPLE_NAME
"#,
        );
        write(
            &sfz_path,
            r#"
#define $SAMPLE_NAME snare.wav
<control>
default_path=samples/
#include "include.sfz"
<region>
hivel=80
"#,
        );

        let regions = parse_soundfont(&sfz_path).unwrap();

        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].velrange, 1..=80);
        assert_eq!(regions[0].sample_path, sample_path.canonicalize().unwrap());

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn parse_soundfont_inherits_parent_group_values() {
        let dir = create_temp_dir("inheritance");
        let sfz_path = dir.join("instrument.sfz");
        let sample_path = dir.join("samples").join("tone.wav");

        write(&sample_path, "");
        write(
            &sfz_path,
            r#"
<global>
lokey=c3
<group>
hikey=d3
volume=-6
<region>
sample=samples/tone.wav
"#,
        );

        let regions = parse_soundfont(&sfz_path).unwrap();

        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].keyrange, 48..=50);
        assert_eq!(regions[0].volume, -6);

        fs::remove_dir_all(dir).unwrap();
    }
}
