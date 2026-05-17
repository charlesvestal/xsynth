// MOVE FORK / 2026-05-16: Mac-side chord-render harness for
// debugging output-stage distortion. Loads an SFZ, plays a fixed
// chord at vel=127, renders ~5 seconds, and writes TWO output WAVs:
//
//   <out>.raw.wav        — pre-master-stage f32 sum from xsynth-core
//                          (no master gain, no limiter, no soft clip).
//                          Shows what the voice mixer produces. Peak
//                          can exceed ±1.0.
//   <out>.processed.wav  — master gain + peak limiter applied
//                          (mirrors the on-device output stage in
//                          xsynth_plugin.c). Peak should be ≤ ±0.95.
//
// Usage:
//   chord_render <sfz_path> <out_basename>
//
// The chord is a C major + octave: keys 60, 64, 67, 72 at vel 127.
// Held for 3 s, then release; total ~5 s of audio captured.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use xsynth_core::{
    channel::{ChannelAudioEvent, ChannelConfigEvent, ChannelEvent, ChannelInitOptions},
    channel_group::{
        ChannelGroup, ChannelGroupConfig, ParallelismOptions, SynthEvent, SynthFormat,
    },
    soundfont::{Interpolator, SampleSoundfont, SoundfontBase, SoundfontInitOptions},
    AudioPipe, AudioStreamParams, ChannelCount,
};

const SAMPLE_RATE: u32 = 44100;
const FRAMES_PER_BLOCK: usize = 128;
const STEREO: usize = 2;

const CHORD_KEYS: &[u8] = &[60, 64, 67, 72]; // C major + octave
const CHORD_VEL: u8 = 127;
const SUSTAIN_SECS: f32 = 3.0;
const RELEASE_TAIL_SECS: f32 = 2.0;
/// Mirrors xsynth_plugin.c's master gain default. Kept at 0.7 to
/// preserve prior preset loudness; the soft-clip handles peaks.
const MASTER_GAIN: f32 = 0.7;
/// Stateless soft-clip knee. Linear below; tanh-shaped above.
const SOFTCLIP_KNEE: f32 = 0.9;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("Usage: chord_render <sfz_path> <out_basename>");
        std::process::exit(1);
    }
    let sfz_path = PathBuf::from(&args[0]);
    let out_base = &args[1];

    let stream_params = AudioStreamParams::new(SAMPLE_RATE, ChannelCount::Stereo);

    eprintln!("[chord_render] loading SFZ: {}", sfz_path.display());
    let sf = SampleSoundfont::new_sfz(
        sfz_path,
        stream_params,
        SoundfontInitOptions {
            bank: None,
            preset: None,
            vol_envelope_options: Default::default(),
            use_effects: true,
            interpolator: Interpolator::Linear,
        },
    )
    .expect("failed to load SFZ");

    let cfg = ChannelGroupConfig {
        channel_init_options: ChannelInitOptions { fade_out_killing: true },
        format: SynthFormat::Custom { channels: 1 },
        audio_params: stream_params,
        parallelism: ParallelismOptions::AUTO_PER_CHANNEL,
    };
    let mut group = ChannelGroup::new(cfg);

    let sf_arc: Arc<dyn SoundfontBase> = Arc::new(sf);
    group.send_event(SynthEvent::Channel(
        0,
        ChannelEvent::Config(ChannelConfigEvent::SetSoundfonts(vec![sf_arc])),
    ));

    // Pause briefly so any async setup settles.
    std::thread::sleep(Duration::from_millis(50));

    // Fire the chord NoteOns all at once at t=0.
    eprintln!("[chord_render] noteon chord: {:?} @ vel {}", CHORD_KEYS, CHORD_VEL);
    for &k in CHORD_KEYS {
        group.send_event(SynthEvent::Channel(
            0,
            ChannelEvent::Audio(ChannelAudioEvent::NoteOn { key: k, vel: CHORD_VEL }),
        ));
    }

    let total_frames =
        ((SUSTAIN_SECS + RELEASE_TAIL_SECS) * SAMPLE_RATE as f32) as usize;
    let sustain_frames = (SUSTAIN_SECS * SAMPLE_RATE as f32) as usize;
    let total_blocks = (total_frames + FRAMES_PER_BLOCK - 1) / FRAMES_PER_BLOCK;
    let sustain_blocks = sustain_frames / FRAMES_PER_BLOCK;

    let mut raw_block = vec![0.0f32; FRAMES_PER_BLOCK * STEREO];
    let mut raw_all: Vec<f32> = Vec::with_capacity(total_blocks * FRAMES_PER_BLOCK * STEREO);
    let mut processed_all: Vec<i16> = Vec::with_capacity(total_blocks * FRAMES_PER_BLOCK * STEREO);
    let mut released = false;

    let mut peak_raw: f32 = 0.0;

    for block in 0..total_blocks {
        if block == sustain_blocks && !released {
            eprintln!("[chord_render] noteoff at block {}", block);
            for &k in CHORD_KEYS {
                group.send_event(SynthEvent::Channel(
                    0,
                    ChannelEvent::Audio(ChannelAudioEvent::NoteOff { key: k }),
                ));
            }
            released = true;
        }

        raw_block.iter_mut().for_each(|s| *s = 0.0);
        group.read_samples(&mut raw_block);

        for &v in raw_block.iter() {
            let av = v.abs();
            if av > peak_raw { peak_raw = av; }
        }
        raw_all.extend_from_slice(&raw_block);

        // Apply master gain + stateless soft-clip (mirrors xsynth_plugin.c v2).
        let space = 1.0 - SOFTCLIP_KNEE;
        for i in 0..FRAMES_PER_BLOCK {
            let mut l = raw_block[i * 2] * MASTER_GAIN;
            let mut r = raw_block[i * 2 + 1] * MASTER_GAIN;
            let al = l.abs();
            if al > SOFTCLIP_KNEE {
                let over = al - SOFTCLIP_KNEE;
                let shaped = SOFTCLIP_KNEE + space * (over / space).tanh();
                l = if l < 0.0 { -shaped } else { shaped };
            }
            let ar = r.abs();
            if ar > SOFTCLIP_KNEE {
                let over = ar - SOFTCLIP_KNEE;
                let shaped = SOFTCLIP_KNEE + space * (over / space).tanh();
                r = if r < 0.0 { -shaped } else { shaped };
            }
            processed_all.push((l * 32767.0) as i16);
            processed_all.push((r * 32767.0) as i16);
        }
    }

    // Compute peaks of processed output.
    let mut peak_proc: i16 = 0;
    for &v in processed_all.iter() {
        let av = v.unsigned_abs() as i16;
        if av > peak_proc { peak_proc = av; }
    }

    eprintln!(
        "[chord_render] raw f32 peak (pre-master): {:.4} ({:+.2} dBFS)",
        peak_raw,
        20.0 * peak_raw.max(1e-6).log10()
    );
    eprintln!(
        "[chord_render] processed i16 peak: {} ({:.1}% FS, {:+.2} dBFS)",
        peak_proc,
        peak_proc as f32 / 32768.0 * 100.0,
        20.0 * (peak_proc as f32 / 32768.0).max(1e-6).log10()
    );

    write_wav_f32(&format!("{}.raw.wav", out_base), &raw_all, SAMPLE_RATE);
    write_wav_i16(&format!("{}.processed.wav", out_base), &processed_all, SAMPLE_RATE);
    eprintln!("[chord_render] wrote {}.raw.wav + {}.processed.wav", out_base, out_base);
}

/// Write interleaved stereo f32 samples as 32-bit float WAV.
fn write_wav_f32(path: &str, samples: &[f32], sample_rate: u32) {
    use std::io::Write;
    let n_chans = 2u16;
    let bits = 32u16;
    let byte_rate = sample_rate * n_chans as u32 * 4;
    let block_align = n_chans * 4;
    let data_size = (samples.len() * 4) as u32;
    let mut f = std::fs::File::create(path).expect("create wav");
    // RIFF/WAVE header — fmt chunk format 3 (IEEE float).
    f.write_all(b"RIFF").unwrap();
    f.write_all(&(36u32 + data_size).to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&3u16.to_le_bytes()).unwrap(); // PCM float
    f.write_all(&n_chans.to_le_bytes()).unwrap();
    f.write_all(&sample_rate.to_le_bytes()).unwrap();
    f.write_all(&byte_rate.to_le_bytes()).unwrap();
    f.write_all(&block_align.to_le_bytes()).unwrap();
    f.write_all(&bits.to_le_bytes()).unwrap();
    f.write_all(b"data").unwrap();
    f.write_all(&data_size.to_le_bytes()).unwrap();
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(samples.as_ptr() as *const u8, samples.len() * 4)
    };
    f.write_all(bytes).unwrap();
}

/// Write interleaved stereo i16 samples as 16-bit PCM WAV.
fn write_wav_i16(path: &str, samples: &[i16], sample_rate: u32) {
    use std::io::Write;
    let n_chans = 2u16;
    let bits = 16u16;
    let byte_rate = sample_rate * n_chans as u32 * 2;
    let block_align = n_chans * 2;
    let data_size = (samples.len() * 2) as u32;
    let mut f = std::fs::File::create(path).expect("create wav");
    f.write_all(b"RIFF").unwrap();
    f.write_all(&(36u32 + data_size).to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
    f.write_all(&n_chans.to_le_bytes()).unwrap();
    f.write_all(&sample_rate.to_le_bytes()).unwrap();
    f.write_all(&byte_rate.to_le_bytes()).unwrap();
    f.write_all(&block_align.to_le_bytes()).unwrap();
    f.write_all(&bits.to_le_bytes()).unwrap();
    f.write_all(b"data").unwrap();
    f.write_all(&data_size.to_le_bytes()).unwrap();
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(samples.as_ptr() as *const u8, samples.len() * 2)
    };
    f.write_all(bytes).unwrap();
}
