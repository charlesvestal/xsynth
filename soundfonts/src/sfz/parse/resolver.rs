use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use encoding_rs::UTF_8;
use encoding_rs_io::DecodeReaderBytesBuilder;

use super::{
    defines::apply_defines, grammar_token_into_sfz_token, ErrorTolerantToken, SfzParseError,
    SfzToken, SfzTokenWithMeta,
};

#[derive(Default)]
pub(super) struct TokenResolver {
    defines: HashMap<String, String>,
    include_stack: Vec<PathBuf>,
    include_cache: HashMap<(PathBuf, usize), Vec<SfzToken>>,
    define_generation: usize,
}

impl TokenResolver {
    pub(super) fn resolve_file(
        &mut self,
        file_path: &Path,
    ) -> Result<Vec<SfzToken>, SfzParseError> {
        let file_path = file_path
            .canonicalize()
            .map_err(|_| SfzParseError::FailedToReadFile(file_path.to_owned()))?;

        if self.include_stack.contains(&file_path) {
            return Err(SfzParseError::IncludeCycle(file_path));
        }

        self.include_stack.push(file_path.clone());
        let result = self.resolve_canonical_file(&file_path);
        self.include_stack.pop();
        result
    }

    fn resolve_canonical_file(&mut self, file_path: &Path) -> Result<Vec<SfzToken>, SfzParseError> {
        let file = read_file(file_path)?;

        // The path is canonicalized above, so it always has a parent directory.
        let parent_path = file_path.parent().unwrap();
        let mut tokens = Vec::new();

        for token in ErrorTolerantToken::parse_as_iter(&file) {
            let token = token?;
            let Some(token) = grammar_token_into_sfz_token(token, &self.defines)? else {
                continue;
            };

            match token {
                SfzTokenWithMeta::Import(path) => {
                    let path = apply_defines(&path, &self.defines).into_owned();
                    // MOVE FORK / 2026-05-16: SFZ spec says #include resolves
                    // relative to the file containing the directive. In
                    // practice many SFZ libraries (e.g. Salamander Grand
                    // Piano V3) emit includes inside nested sub-files
                    // using paths relative to the SFZ ROOT — Salamander's
                    // Data/notes.txt contains `#include "Data/vel_01.txt"`
                    // which the author intended to resolve to
                    // <root>/Data/vel_01.txt, not <root>/Data/Data/vel_01.txt.
                    // sfizz and ARIA both fall back to root-relative when
                    // the direct relative path doesn't exist; we match
                    // that convention to keep author-shipped libraries
                    // loadable without manual fixups.
                    let direct = parent_path.join(&path);
                    let include_path = if direct.exists() {
                        direct
                    } else if let Some(root) = self.include_stack.first() {
                        let root_parent = root.parent().unwrap_or(root);
                        let root_relative = root_parent.join(&path);
                        if root_relative.exists() {
                            root_relative
                        } else {
                            direct  // surface the spec-relative path in error
                        }
                    } else {
                        direct
                    };
                    let cache_key = (include_path.clone(), self.define_generation);

                    if let Some(cached_tokens) = self.include_cache.get(&cache_key) {
                        tokens.extend_from_slice(cached_tokens);
                        continue;
                    }

                    let resolved = self.resolve_file(&include_path)?;
                    self.include_cache.insert(cache_key, resolved.clone());
                    tokens.extend(resolved);
                }
                SfzTokenWithMeta::Group(group) => tokens.push(SfzToken::Group(group)),
                SfzTokenWithMeta::Opcode(opcode) => tokens.push(SfzToken::Opcode(opcode)),
                SfzTokenWithMeta::Define(variable, value) => {
                    self.define_generation += 1;
                    self.include_cache.clear();
                    self.defines
                        .insert(variable.trim().to_owned(), value.trim().to_owned());
                }
            }
        }

        Ok(tokens)
    }
}

fn read_file(file_path: &Path) -> Result<String, SfzParseError> {
    let file =
        File::open(file_path).map_err(|_| SfzParseError::FailedToReadFile(file_path.to_owned()))?;
    let mut reader = BufReader::new(
        DecodeReaderBytesBuilder::new()
            .encoding(Some(UTF_8))
            .build(file),
    );
    let mut file = String::new();
    reader
        .read_to_string(&mut file)
        .map_err(|_| SfzParseError::FailedToReadFile(file_path.to_owned()))?;
    Ok(file)
}
