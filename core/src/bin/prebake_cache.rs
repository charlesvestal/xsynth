// MOVE FORK / 2026-05-16: pre-bake .x44c sample caches off-device.
//
// Walks one or more directories for audio files (.flac / .wav / .ogg /
// .aiff) and decodes each into the `.x44c` cache format that xsynth's
// streaming loader uses on Move. Designed to run on Mac (or any
// faster-than-Move machine) so commercial sample libraries can be
// converted in minutes instead of the 30-60+ minutes the same work
// takes on Move's microSD-backed CPU.
//
// Usage:
//   prebake_cache <dir> [<dir> ...]
//
// Output: writes <sample>.x44c next to every audio file. Subsequent
// rsync of the `.x44c` files into the Move's sample tree skips the
// decode entirely.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use xsynth_core::soundfont::prebake_audio_cache;
use xsynth_core::{AudioStreamParams, ChannelCount};

const TARGET_RATE: u32 = 44_100;

/// Decoded samples are held in heap as i16 PCM (channels × frames × 2 B).
/// A 30s, 48kHz/24-bit/stereo source decodes to ~5.5 MB; some piano libraries
/// (Claustrophobic Piano V2, Hunter's Ampex) ship 90s+ samples that peak at
/// ~750 MB per decode. Move only has 1.8 GB RAM total and the host audio
/// engine sits at ~700 MB, so concurrent decodes OOM the device.
///
/// `PREBAKE_THREADS` env var controls the rayon pool size; default 1 is safe
/// on Move. Set to `0` for "use all cores" (fine on Mac).
const ENV_THREADS: &str = "PREBAKE_THREADS";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: prebake_cache <dir> [<dir> ...]");
        eprintln!();
        eprintln!("Pre-bakes .x44c caches for every audio file found");
        eprintln!("recursively under the given directories. Output files");
        eprintln!("land next to the source audio files.");
        std::process::exit(1);
    }

    let mut files: Vec<PathBuf> = Vec::new();
    for arg in &args {
        let root = PathBuf::from(arg);
        if !root.exists() {
            eprintln!("warning: {} does not exist; skipping", root.display());
            continue;
        }
        walk(&root, &mut files);
    }
    if files.is_empty() {
        eprintln!("No audio files found.");
        std::process::exit(0);
    }

    let threads: usize = std::env::var(ENV_THREADS)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    eprintln!(
        "Pre-baking {} files into .x44c caches @ {} Hz stereo ({} thread{})",
        files.len(),
        TARGET_RATE,
        if threads == 0 { "all".into() } else { threads.to_string() },
        if threads == 1 { "" } else { "s" }
    );

    let stream_params = AudioStreamParams::new(TARGET_RATE, ChannelCount::Stereo);
    let counter = AtomicUsize::new(0);
    let errors = AtomicUsize::new(0);
    let total = files.len();
    let start = Instant::now();

    files.par_iter().for_each(|path| {
        let result = prebake_audio_cache(path, stream_params);
        let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
        match result {
            Ok(()) => {}
            Err(e) => {
                errors.fetch_add(1, Ordering::Relaxed);
                eprintln!("FAIL [{}] {}: {:?}", n, path.display(), e);
            }
        }
        if n == 1 || n == total || n % 16 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = n as f64 / elapsed.max(0.001);
            let eta = (total - n) as f64 / rate.max(0.001);
            eprintln!(
                "[{}/{}] {:.1}s elapsed, {:.1} files/s, ETA {:.1}s",
                n, total, elapsed, rate, eta
            );
        }
    });

    let elapsed = start.elapsed().as_secs_f64();
    let err_n = errors.load(Ordering::Relaxed);
    eprintln!(
        "Done: {}/{} succeeded in {:.1}s ({:.1} files/s)",
        total - err_n,
        total,
        elapsed,
        total as f64 / elapsed.max(0.001),
    );
    if err_n > 0 {
        std::process::exit(2);
    }
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    if dir.is_file() {
        if is_audio_ext(dir) {
            out.push(dir.to_path_buf());
        }
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else if is_audio_ext(&path) {
            out.push(path);
        }
    }
}

fn is_audio_ext(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_lowercase)
            .as_deref(),
        Some("flac") | Some("wav") | Some("ogg") | Some("aiff") | Some("aif")
    )
}
