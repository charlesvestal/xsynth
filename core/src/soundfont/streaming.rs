// MOVE FORK / 2026-05-16: disk-streaming sample backend.
//
// xsynth's default sample model is "fully resident in RAM" — every decoded
// sample lives as Arc<[i16]> (heap) or mmap'd cache (since this fork). That
// breaks for libraries larger than Move's RAM (Salamander Grand Piano:
// ~1.1 GB of decoded i16). This module adds a third path: stream samples
// from disk on a background thread, expose a per-voice ring buffer to the
// audio thread, never block the audio thread on I/O.
//
// Architecture:
//   - StreamedSampleSource: shared per-sample. Owns the cache file handle,
//     metadata, and a small "head" buffer (first ~100 ms of every channel,
//     always resident). Heads are kept in RAM so the very first samples
//     read after NoteOn are fault-free.
//   - StreamRing: per-voice sliding window over the streamed sample. Owns
//     a fixed-capacity i16 buffer. Voice reads from ring (audio thread);
//     I/O pool writes to ring (background thread). Atomic positions
//     bound the valid window.
//   - IoPool: one background thread per channel-group / soundfont. Holds
//     a registry of live rings (via Weak<RingShared>), periodically scans
//     and refills any that fall below a low-water mark.
//
// Day 1 scope: data structures + IoPool plumbing, no spawner wiring yet.
// Spawner integration is day 2; loop handling is day 3.

use std::cell::UnsafeCell;
use std::fs::File;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, Sender};

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{Decoder, DecoderOptions};
use symphonia::core::conv::IntoSample;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::sample::Sample;

/// MOVE FORK / 2026-05-16: process-global underrun counter. Every time a
/// VoiceStream::get() falls past the head buffer and hits an unfilled
/// ring slot, this counter ticks. Audio glitches under chord-burst spawn
/// pressure show up here. The plugin's perf log snapshots+resets this
/// per render block to correlate underruns with concurrent voice counts.
fn underrun_counter() -> &'static AtomicU32 {
    use std::sync::OnceLock;
    static C: OnceLock<AtomicU32> = OnceLock::new();
    C.get_or_init(|| AtomicU32::new(0))
}

/// MOVE FORK / 2026-05-17 diag: split underrun categories.
/// - ahead: pos >= write_pos AND not EOF (ring not yet filled to here)
/// - behind: pos < write_pos - RING_FRAMES (voice fell behind window —
///   typically loop-wrap or pitch glitch)
/// - before_head: pos < head_start (pathological — voice read before head's
///   start file frame)
fn underrun_ahead_counter() -> &'static AtomicU32 {
    use std::sync::OnceLock;
    static C: OnceLock<AtomicU32> = OnceLock::new();
    C.get_or_init(|| AtomicU32::new(0))
}
fn underrun_behind_counter() -> &'static AtomicU32 {
    use std::sync::OnceLock;
    static C: OnceLock<AtomicU32> = OnceLock::new();
    C.get_or_init(|| AtomicU32::new(0))
}
fn underrun_before_head_counter() -> &'static AtomicU32 {
    use std::sync::OnceLock;
    static C: OnceLock<AtomicU32> = OnceLock::new();
    C.get_or_init(|| AtomicU32::new(0))
}

/// Public accessor: snapshot + reset the underrun counter. Called from
/// the C plugin's render-block perf log so underruns appear next to
/// voice count, render time, etc.
pub fn take_underrun_count() -> u32 {
    underrun_counter().swap(0, Ordering::Relaxed)
}

/// MOVE FORK / 2026-05-17 diag: per-category underrun snapshots.
/// Returns (ahead, behind, before_head). Each call resets its counter.
pub fn take_underrun_breakdown() -> (u32, u32, u32) {
    (
        underrun_ahead_counter().swap(0, Ordering::Relaxed),
        underrun_behind_counter().swap(0, Ordering::Relaxed),
        underrun_before_head_counter().swap(0, Ordering::Relaxed),
    )
}

/// Always-resident head buffer length in frames. 1000 ms at 44.1 kHz.
///
/// Sized to absorb the I/O thread's serial pread backlog under chord-
/// burst spawn pressure. Each freshly-spawned voice starts with an empty
/// ring buffer; the I/O thread services rings sequentially at 10–30 ms
/// per pread on Move's microSD under contention. An 8-note stereo
/// chord on a multi-region preset (Legacy Knight: 38 regions, ~2
/// voices/NoteOn) spawns ~32 rings simultaneously, producing 600–1000 ms
/// of cumulative I/O backlog. 500 ms head was insufficient — voices in
/// the trailing half of the spawn order underran their ring buffer
/// before refill landed, producing audible discontinuities. 1 s head
/// covers the worst-case chord-burst backlog cleanly.
///
/// RAM cost: HEAD_FRAMES × n_chans × 2 bytes per sample. For Salamander
/// (641 stereo samples): ~113 MB head resident. Within Move's 2 GB
/// RAM budget; <10% of total system memory.
///
/// Tradeoff vs longer head: preset load time grows linearly with head
/// size (each sample reads HEAD_FRAMES from disk at load). 1 s head
/// adds ~4 s to Salamander cold-load vs 100 ms head; still <10 s total.
///
/// Future work if chord-bursts grow heavier (e.g. 16-note multi-layer):
/// parallel I/O workers — 2 threads servicing the same registry halves
/// the backlog. Day 4.
pub const HEAD_FRAMES: usize = 44100;

/// Per-voice ring buffer capacity in frames. 32768 = ~744 ms at 44.1 kHz.
/// Sized to give the I/O worker comfortable headroom between refills
/// even under chord-burst conditions where the worker is also doing
/// initial fills of freshly registered rings.
///
/// MUST be a power of 2 — index wraparound uses `pos & (CAP - 1)`.
pub const RING_FRAMES: usize = 32768;
const _: () = assert!(RING_FRAMES.is_power_of_two());

/// Chunk size for each disk read (frames per channel). 16384 = ~372 ms.
///
/// MOVE FORK / 2026-05-16: bumped 1024 → 8192 → 16384 over two
/// iterations. Per-pread syscall + microSD controller overhead
/// dominates small reads; larger chunks amortize the overhead. Per
/// ring at 16384 frames per refill: one refill every ~372 ms of voice
/// playback. Worker handling 14 rings does 14 refills per ~84 ms
/// (~6 ms/pread under chord-burst contention), so per-ring fill rate
/// is ~3.7× drain rate. Margin handles concurrent fresh-ring
/// registrations + microSD GC tail latency without underrun.
const REFILL_CHUNK_FRAMES: usize = 16384;

/// Low-water mark: when `write_pos - consumed_hint` drops below this many
/// frames, the I/O pool schedules a refill. Set to RING_FRAMES/2 so a
/// refill of REFILL_CHUNK_FRAMES (=RING_FRAMES/2) always fits without
/// overrunning the consumer.
const LOW_WATER_FRAMES: usize = RING_FRAMES / 2;

/// Number of I/O worker threads in the pool. Each owns its own working
/// set; new ring registrations are round-robin distributed. 3 workers
/// give ~50% more aggregate refill throughput vs 2; needed to fully
/// absorb chord-burst spawn bursts on multi-layer presets with sustain
/// pedal held (Legacy Knight granular: ~28 sustained voices + 6-12
/// fresh rings per noteon). Move has 4 CPU cores so 3 I/O threads +
/// audio thread + xsynth rayon pool still leaves headroom.
const N_IO_WORKERS: usize = 3;

/// On-disk layout of sample data within the source file. Determines how
/// the I/O pool reads bytes for a given (channel, frame) pair.
#[derive(Debug, Clone)]
pub enum SampleLayout {
    /// `.x44c` cache: channels stored contiguously, ch0 frames first then
    /// ch1 frames. Reading channel C frame F = file byte
    /// `byte_offset_per_channel[C] + F * 2` (i16).
    Separated {
        byte_offset_per_channel: Vec<u64>,
    },
    /// Canonical interleaved PCM (e.g. 16-bit WAV): each frame is N
    /// channels of i16 packed sequentially. Reading channel C frame F =
    /// file byte `data_byte_offset + F * frame_bytes + C * 2`. The IoPool
    /// reads `count * frame_bytes` bytes per refill and strides through
    /// to extract the requested channel.
    Interleaved {
        data_byte_offset: u64,
        frame_bytes: u32,
    },
    /// MOVE FORK / 2026-05-16: streamed FLAC. The IoPool opens a
    /// per-ring symphonia decoder, seeks, and decodes packets on demand.
    /// No `.x44c` cache needed — works direct from the .flac file.
    /// Only valid when sample_rate == target_rate (no SRC; day 3 work).
    Flac {
        /// Path to the .flac file. Each ring's worker opens its own
        /// File handle + decoder so reads don't conflict on shared
        /// seek positions.
        path: std::path::PathBuf,
    },
}

/// One streamed-sample source. Shared (Arc) across all voices that play
/// this sample. Provides the file handle + layout metadata. Heads are
/// per-(spawner-region, offset), NOT per-sample — see `read_head_at()`.
pub struct StreamedSampleSource {
    /// Open file handle. pread'd by the I/O pool; shared across all
    /// voices/channels of this sample. Held in Arc so the FD stays open
    /// while any voice references the source.
    pub file: Arc<File>,
    /// On-disk layout. Determines the pread byte-offset formula.
    pub layout: SampleLayout,
    /// Total frame count per channel, in `data_rate` frames.
    pub frames: usize,
    /// Number of audio channels (1 mono, 2 stereo).
    pub n_chans: usize,
    /// Source sample rate (Hz). The natural rate of the original audio
    /// file — used by SFZ frame-index conversion (`region.offset` is
    /// expressed in source-rate frames per spec).
    pub src_rate: u32,
    /// MOVE FORK / 2026-05-17 (day 3): the sample rate of data actually
    /// stored in the ring buffer (and head buffer).
    /// - `.x44c` cache (Separated): data is pre-resampled to target rate,
    ///   so `data_rate == cached_tgt_rate == stream_params.sample_rate`.
    /// - WAV-direct (Interleaved): data stays in the WAV file's rate, so
    ///   `data_rate == src_rate`.
    /// - FLAC-direct (Flac): data is decoded at the FLAC's native rate
    ///   (no I/O-thread resampling), so `data_rate == src_rate`.
    ///
    /// When `data_rate != stream_params.sample_rate`, the voice spawner
    /// multiplies its `speed_mult` by `data_rate / stream_params.sample_rate`
    /// so the existing linear-interp playback rate-converts on the fly.
    pub data_rate: u32,
}

impl StreamedSampleSource {
    /// Bytes per i16 sample frame in the per-channel rings (always 2;
    /// rings store i16 regardless of source layout).
    pub const BYTES_PER_RING_FRAME: u64 = 2;

    /// MOVE FORK / 2026-05-16: pre-load a resident head buffer per
    /// channel starting at `frame_offset` frames into the file. Used at
    /// SFZ load time to populate per-region heads — each unique
    /// (sample_path, region.offset) pair gets one head, deduplicated
    /// across regions. Returns None on read failure (caller falls back
    /// to a zero-filled head so the voice gets silence at the start
    /// rather than crashing).
    /// MOVE FORK / 2026-05-17: convenience wrapper that calls
    /// `read_head_at_len(frame_offset, HEAD_FRAMES)`. Kept for callers
    /// that don't need a custom head length.
    #[cfg(unix)]
    pub fn read_head_at(&self, frame_offset: usize) -> Option<Vec<Arc<[i16]>>> {
        self.read_head_at_len(frame_offset, HEAD_FRAMES)
    }

    /// MOVE FORK / 2026-05-17: pre-load a resident head buffer of
    /// `desired_len` frames starting at `frame_offset`. Looped SFZ
    /// regions extend this past HEAD_FRAMES so the entire loop region
    /// is always resident — the bounded ring buffer can't satisfy a
    /// loop-wrap read that lands earlier than (write_pos - RING_FRAMES).
    #[cfg(unix)]
    pub fn read_head_at_len(
        &self,
        frame_offset: usize,
        desired_len: usize,
    ) -> Option<Vec<Arc<[i16]>>> {
        let frames_available = self.frames.saturating_sub(frame_offset);
        let head_len = desired_len.min(frames_available);
        if head_len == 0 {
            // Region offset is at or past the end of the sample — give
            // empty heads; voice will read silence from the ring.
            return Some((0..self.n_chans).map(|_| Arc::from(Vec::new().into_boxed_slice())).collect());
        }
        let mut heads = Vec::with_capacity(self.n_chans);
        match &self.layout {
            SampleLayout::Separated { byte_offset_per_channel } => {
                for c in 0..self.n_chans {
                    let mut buf: Vec<i16> = vec![0; head_len];
                    let byte_view = unsafe {
                        std::slice::from_raw_parts_mut(
                            buf.as_mut_ptr() as *mut u8,
                            head_len * 2,
                        )
                    };
                    let byte_off = byte_offset_per_channel[c] + (frame_offset as u64) * 2;
                    let n = self.file.read_at(byte_view, byte_off).ok()?;
                    if n < head_len * 2 {
                        return None;
                    }
                    heads.push(Arc::from(buf.into_boxed_slice()));
                }
            }
            SampleLayout::Interleaved { data_byte_offset, frame_bytes } => {
                let fb = *frame_bytes as usize;
                let byte_off = data_byte_offset + (frame_offset as u64) * (*frame_bytes as u64);
                let byte_len = head_len * fb;
                let mut interleaved = vec![0u8; byte_len];
                let n = self.file.read_at(&mut interleaved, byte_off).ok()?;
                if n < byte_len {
                    return None;
                }
                for c in 0..self.n_chans {
                    let mut buf: Vec<i16> = Vec::with_capacity(head_len);
                    let ch_byte_offset = c * 2;
                    for f in 0..head_len {
                        let off = f * fb + ch_byte_offset;
                        buf.push(i16::from_le_bytes([interleaved[off], interleaved[off + 1]]));
                    }
                    heads.push(Arc::from(buf.into_boxed_slice()));
                }
            }
            SampleLayout::Flac { path } => {
                // Open a fresh per-call symphonia decoder. Heads are read
                // once at SFZ load time, so the cost of decoder setup is
                // amortized across the preset lifetime.
                let (mut format, mut decoder, track_id) =
                    open_flac_decoder(path)?;
                // Seek to frame_offset. Symphonia's FLAC reader supports
                // accurate seeks; actual_ts may land before required_ts
                // and we skip leading samples until aligned.
                let seeked = format
                    .seek(
                        SeekMode::Accurate,
                        SeekTo::TimeStamp {
                            ts: frame_offset as u64,
                            track_id,
                        },
                    )
                    .ok()?;
                let mut cur_frame = seeked.actual_ts as usize;
                let target_start = frame_offset;
                let target_end = frame_offset + head_len;
                let mut per_chan: Vec<Vec<i16>> =
                    (0..self.n_chans).map(|_| Vec::with_capacity(head_len)).collect();
                while cur_frame < target_end {
                    let packet = match format.next_packet() {
                        Ok(p) => p,
                        Err(_) => break,
                    };
                    if packet.track_id() != track_id {
                        continue;
                    }
                    let audio = match decoder.decode(&packet) {
                        Ok(a) => a,
                        Err(SymphoniaError::DecodeError(_)) => continue,
                        Err(_) => break,
                    };
                    let packet_frames = packet_frame_count(&audio);
                    let packet_start = cur_frame;
                    let packet_end = packet_start + packet_frames;
                    // Window: intersect [packet_start, packet_end) with [target_start, target_end).
                    let copy_from = target_start.max(packet_start);
                    let copy_to = target_end.min(packet_end);
                    if copy_from < copy_to {
                        let local_start = copy_from - packet_start;
                        let local_end = copy_to - packet_start;
                        for c in 0..self.n_chans {
                            extract_packet_channel_i16(
                                &audio, c, local_start, local_end, &mut per_chan[c],
                            );
                        }
                    }
                    cur_frame = packet_end;
                }
                for c in 0..self.n_chans {
                    if per_chan[c].len() < head_len {
                        // Short read (EOF or decode error mid-head). Pad
                        // with silence so the voice still gets a fault-
                        // free starting window.
                        per_chan[c].resize(head_len, 0);
                    } else if per_chan[c].len() > head_len {
                        per_chan[c].truncate(head_len);
                    }
                    heads.push(Arc::from(per_chan[c].clone().into_boxed_slice()));
                }
            }
        }
        Some(heads)
    }
}

/// MOVE FORK / 2026-05-17 (day 2): open a fresh symphonia FLAC format
/// reader + decoder pointed at `path`. Each voice's ring gets its own
/// decoder so seek state doesn't collide. Used by both the head-read
/// path (one-shot per load) and the I/O pool's per-ring refill loop.
fn open_flac_decoder(
    path: &std::path::Path,
) -> Option<(Box<dyn FormatReader>, Box<dyn Decoder>, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("flac");
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .ok()?;
    let format = probed.format;
    let (track_id, codec_params) = {
        let track = format.default_track()?;
        (track.id, track.codec_params.clone())
    };
    let decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .ok()?;
    Some((format, decoder, track_id))
}

/// Frame count of a decoded audio buffer ref.
fn packet_frame_count(buf: &AudioBufferRef) -> usize {
    match buf {
        AudioBufferRef::U8(b) => b.chan(0).len(),
        AudioBufferRef::U16(b) => b.chan(0).len(),
        AudioBufferRef::U24(b) => b.chan(0).len(),
        AudioBufferRef::U32(b) => b.chan(0).len(),
        AudioBufferRef::S8(b) => b.chan(0).len(),
        AudioBufferRef::S16(b) => b.chan(0).len(),
        AudioBufferRef::S24(b) => b.chan(0).len(),
        AudioBufferRef::S32(b) => b.chan(0).len(),
        AudioBufferRef::F32(b) => b.chan(0).len(),
        AudioBufferRef::F64(b) => b.chan(0).len(),
    }
}

/// Append `[local_start..local_end)` of channel `c` from `buf` (converted
/// to i16) to `out`. Sample-format-agnostic via Symphonia's IntoSample.
fn extract_packet_channel_i16(
    buf: &AudioBufferRef,
    c: usize,
    local_start: usize,
    local_end: usize,
    out: &mut Vec<i16>,
) {
    fn push<S: Sample + IntoSample<i16>>(
        chan: &[S],
        local_start: usize,
        local_end: usize,
        out: &mut Vec<i16>,
    ) {
        let end = local_end.min(chan.len());
        if local_start >= end {
            return;
        }
        out.reserve(end - local_start);
        for &s in &chan[local_start..end] {
            out.push(s.into_sample());
        }
    }
    match buf {
        AudioBufferRef::U8(b)  => push(b.chan(c), local_start, local_end, out),
        AudioBufferRef::U16(b) => push(b.chan(c), local_start, local_end, out),
        AudioBufferRef::U24(b) => push(b.chan(c), local_start, local_end, out),
        AudioBufferRef::U32(b) => push(b.chan(c), local_start, local_end, out),
        AudioBufferRef::S8(b)  => push(b.chan(c), local_start, local_end, out),
        AudioBufferRef::S16(b) => push(b.chan(c), local_start, local_end, out),
        AudioBufferRef::S24(b) => push(b.chan(c), local_start, local_end, out),
        AudioBufferRef::S32(b) => push(b.chan(c), local_start, local_end, out),
        AudioBufferRef::F32(b) => push(b.chan(c), local_start, local_end, out),
        AudioBufferRef::F64(b) => push(b.chan(c), local_start, local_end, out),
    }
}

impl std::fmt::Debug for StreamedSampleSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StreamedSampleSource({} chans, {} frames)", self.n_chans, self.frames)
    }
}

/// Per-voice sliding-window ring over one channel of a streamed sample.
///
/// Layout: `data[i]` holds frame index `i + (write_pos - data.len())` ...
/// no — using absolute frame indices everywhere, then `pos % capacity` for
/// the array index. Cleaner, no bookkeeping of base position.
///
/// Memory ordering:
/// - I/O thread writes `data[pos % cap] = sample; write_pos.fetch_add(1, Release)`
/// - Voice (audio thread) loads `write_pos` with Acquire to see all writes
///   up to that position.
/// - Voice updates `consumed_hint` with Release; I/O thread loads it with
///   Acquire and uses the gap (write_pos - consumed_hint) to decide refill.
pub struct StreamRing {
    /// Backing storage; capacity is always RING_FRAMES. UnsafeCell so the
    /// I/O thread can write while the audio thread reads — the atomic
    /// positions (write_pos / consumed_hint) gate which slots each side
    /// touches, so no slot is concurrently accessed in practice.
    data: Box<[UnsafeCell<i16>]>,
    /// Next frame index the I/O thread will write. Voice reads positions
    /// in `[write_pos - capacity, write_pos)` from the ring (assuming
    /// they're also >= HEAD_FRAMES; positions below the head boundary
    /// don't use the ring at all).
    write_pos: AtomicUsize,
    /// Voice's high-water consumed position. The I/O thread uses
    /// `write_pos - consumed_hint < LOW_WATER_FRAMES` to gate refill.
    /// The voice updates this on each `get()` so the pool can evict
    /// trailing pages.
    consumed_hint: AtomicUsize,
    /// Set by the I/O thread when EOF is reached (write_pos >= total_frames).
    /// Voice uses this to short-circuit underrun detection past EOF.
    eof: AtomicBool,
    /// MOVE FORK / 2026-05-17 diag: last pos passed to VoiceStream::get
    /// for this ring. Logged when pos goes backward — diagnostic for the
    /// "behind" underrun mystery on no_loop WörliTzer regions.
    last_read_pos: AtomicUsize,
}

impl StreamRing {
    /// MOVE FORK / 2026-05-16: ring is initialized with write_pos =
    /// `head_end_frame`, which equals (head_start + head_len). For
    /// regions with offset=0, head_start=0 and head_end is HEAD_FRAMES
    /// (the historical behavior). For regions with offset>0, the ring
    /// serves the file region just past the per-region head buffer.
    pub fn new_at(head_end_frame: usize) -> Self {
        let data: Vec<UnsafeCell<i16>> =
            (0..RING_FRAMES).map(|_| UnsafeCell::new(0)).collect();
        StreamRing {
            data: data.into_boxed_slice(),
            write_pos: AtomicUsize::new(head_end_frame),
            consumed_hint: AtomicUsize::new(head_end_frame),
            eof: AtomicBool::new(false),
            last_read_pos: AtomicUsize::new(0),
        }
    }

    /// Legacy constructor: ring starts at HEAD_FRAMES (head at offset 0).
    /// Kept for tests; production paths use `new_at`.
    pub fn new() -> Self {
        Self::new_at(HEAD_FRAMES)
    }

    /// Voice-side read at absolute frame `pos`. Returns `None` if the
    /// requested position isn't in the buffered window (underrun) or
    /// is past EOF.
    #[inline]
    pub fn get(&self, pos: usize) -> Option<i16> {
        let write = self.write_pos.load(Ordering::Acquire);
        if pos >= write {
            // Position not yet filled. If EOF, return 0 silently;
            // otherwise this is an underrun (caller decides policy).
            if self.eof.load(Ordering::Relaxed) {
                return Some(0);
            }
            return None;
        }
        // Position is buffered iff it falls within the trailing
        // capacity-sized window of `write`.
        if write.saturating_sub(pos) > RING_FRAMES {
            // Voice rewound past the buffered window (shouldn't happen
            // in normal monotonic playback; only loop-wrap or pitch
            // glitch could trigger). Treat as underrun.
            return None;
        }
        let idx = pos & (RING_FRAMES - 1);
        // SAFETY: idx is bounded by capacity. The Acquire load of
        // write_pos above ensures all writes the I/O thread made before
        // its Release store are visible here, so reading this cell
        // sees the published value. The position invariant means no
        // I/O write is in flight for this idx right now.
        Some(unsafe { *self.data.get_unchecked(idx).get() })
    }

    /// Voice-side hint that frames up to and including `pos` have been
    /// consumed. The I/O thread uses this to gate refills. Idempotent;
    /// monotonically non-decreasing.
    #[inline]
    pub fn mark_consumed(&self, pos: usize) {
        let prev = self.consumed_hint.load(Ordering::Relaxed);
        if pos > prev {
            self.consumed_hint.store(pos, Ordering::Release);
        }
    }

    /// I/O thread query: how many frames can we still write without
    /// overrunning the voice's read pointer (consumed_hint)? Capped at
    /// REFILL_CHUNK_FRAMES so we don't write a whole capacity at once.
    fn refill_budget(&self) -> usize {
        let write = self.write_pos.load(Ordering::Relaxed);
        let consumed = self.consumed_hint.load(Ordering::Acquire);
        // We can fill up to capacity ahead of consumed.
        let max_lead = consumed + RING_FRAMES;
        if write >= max_lead {
            return 0;
        }
        (max_lead - write).min(REFILL_CHUNK_FRAMES)
    }

    /// I/O thread: write a chunk of fresh frames at `write_pos`. Caller
    /// must have queried `refill_budget()` first to know how many frames
    /// to provide. Advances `write_pos` by `chunk.len()` with Release.
    fn write_chunk(&self, chunk: &[i16]) {
        let write_start = self.write_pos.load(Ordering::Relaxed);
        for (i, v) in chunk.iter().enumerate() {
            let idx = (write_start + i) & (RING_FRAMES - 1);
            // SAFETY: idx is bounded by capacity. The voice cannot read
            // this position until we publish via the Release store
            // below (fetch_add on write_pos), so there's no concurrent
            // reader of this slot.
            unsafe {
                *self.data.get_unchecked(idx).get() = *v;
            }
        }
        self.write_pos.fetch_add(chunk.len(), Ordering::Release);
    }

    /// I/O thread: signal EOF (no more frames to fill).
    fn set_eof(&self) {
        self.eof.store(true, Ordering::Release);
    }
}

// SAFETY: StreamRing is shared between the audio thread (reads via get())
// and the I/O thread (writes via write_chunk()). All access goes through
// atomic positions with appropriate ordering; data accesses are gated by
// the atomic positions so no slot is concurrently read+written.
unsafe impl Send for StreamRing {}
unsafe impl Sync for StreamRing {}

/// Voice-facing handle to a registered stream. Holds the ring (read side),
/// the source (for the file handle), the channel index, and a per-voice
/// head buffer pre-loaded starting at `head_start` frames into the file.
///
/// MOVE FORK / 2026-05-16: head is per-VOICE (or per-spawner-region),
/// not per-sample. SFZ regions can have `offset>0` to skip leading
/// silence in the source file; the voice's first reads land at file
/// position `offset`, NOT 0. A per-region head pre-loaded at `offset`
/// keeps the audio thread fault-free at noteon for offset-using
/// regions like Legacy Knight's `42 Gb1.wav` (offset=91550 to skip
/// 2 sec of leading silence). Heads dedupe across regions sharing
/// the same (sample_path, offset) pair.
pub struct VoiceStream {
    pub source: Arc<StreamedSampleSource>,
    pub channel: usize,
    pub ring: Arc<StreamRing>,
    /// Resident head buffer for THIS voice's region. Indexed by
    /// (pos - head_start) for pos in [head_start, head_start + head.len()).
    pub head: Arc<[i16]>,
    /// File frame where this voice's head buffer starts. Usually 0
    /// for regions with no offset, or the region's `offset=` value
    /// for offset-using regions.
    pub head_start: usize,
}

impl VoiceStream {
    /// Voice-side sample read. Routes positions within the head window
    /// to the resident head buffer; everything else to the StreamRing.
    #[inline]
    pub fn get(&self, pos: usize) -> i16 {
        let head_start = self.head_start;
        let head_len = self.head.len();
        if pos >= head_start && pos < head_start + head_len {
            // SAFETY: bounds checked above; head is Arc<[i16]>.
            self.head[pos - head_start]
        } else if pos < head_start {
            // Pathological: voice read a file frame BEFORE the head buffer
            // begins. Only happens if a region's loop wraps to a position
            // before its own offset (rare). Count separately so the perf
            // log surfaces this distinct mode.
            underrun_before_head_counter().fetch_add(1, Ordering::Relaxed);
            underrun_counter().fetch_add(1, Ordering::Relaxed);
            0
        } else {
            // Mark consumed BEFORE reading: lets the I/O pool know how
            // far the voice has advanced, even if this call underruns.
            self.ring.mark_consumed(pos);
            match self.ring.get(pos) {
                Some(v) => v,
                None => {
                    // Categorize the underrun: ahead (ring not filled to
                    // pos yet) vs behind (voice pos fell out of the
                    // trailing window — typically loop wrap).
                    let write = self.ring.write_pos.load(Ordering::Acquire);
                    if pos >= write {
                        underrun_ahead_counter().fetch_add(1, Ordering::Relaxed);
                    } else {
                        underrun_behind_counter().fetch_add(1, Ordering::Relaxed);
                    }
                    underrun_counter().fetch_add(1, Ordering::Relaxed);
                    0
                }
            }
        }
    }

    pub fn length(&self) -> usize {
        self.source.frames
    }
}

// =============================================================================
// I/O Pool
// =============================================================================

/// MOVE FORK / 2026-05-17 (day 2): per-ring FLAC decoder state.
/// One symphonia format reader + decoder per ring (one ring = one
/// channel of one voice), constructed lazily on first refill so registration
/// is cheap. Pending buffer absorbs leftover samples when a decoded packet
/// is larger than the refill budget; reused on the next refill.
struct FlacRingState {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    track_id: u32,
    /// Absolute file frame where the decoder will deliver its NEXT
    /// packet's first sample. Tracked separately from `pending_start`
    /// because the decoder advances by full packets (variable size),
    /// while `pending_start` advances by what we've written to the ring.
    decoder_cursor: usize,
    /// i16 samples for THIS ring's channel that have been decoded but not
    /// yet written to the ring. `pending[0]` corresponds to absolute file
    /// frame `pending_start`.
    pending: Vec<i16>,
    pending_start: usize,
}

/// One ring registration in the pool's working set.
struct RegisteredRing {
    source: Arc<StreamedSampleSource>,
    channel: usize,
    ring: Weak<StreamRing>,
    /// Per-ring FLAC decoder, lazily initialized on first refill iff the
    /// source layout is Flac. None for WAV/Separated layouts.
    flac_state: Option<FlacRingState>,
}

enum PoolCommand {
    /// Add a new ring to the pool's working set.
    Register(RegisteredRing),
    /// Shut down the pool thread.
    Shutdown,
}

/// Background I/O pool. N_IO_WORKERS threads, each owning its own
/// working set; new ring registrations are round-robin distributed.
/// Multiple workers ~double the refill throughput vs single-threaded
/// — critical for sustaining 30+ simultaneous voice streams.
pub struct IoPool {
    tx_workers: Vec<Sender<PoolCommand>>,
    handles: Vec<JoinHandle<()>>,
    next_worker: AtomicUsize,
}

impl IoPool {
    pub fn new() -> Arc<Self> {
        let mut tx_workers = Vec::with_capacity(N_IO_WORKERS);
        let mut handles = Vec::with_capacity(N_IO_WORKERS);
        for i in 0..N_IO_WORKERS {
            let (tx, rx) = bounded::<PoolCommand>(256);
            let handle = thread::Builder::new()
                .name(format!("xsynth-io-pool-{}", i))
                .spawn(move || pool_thread(rx))
                .expect("failed to spawn xsynth-io-pool thread");
            tx_workers.push(tx);
            handles.push(handle);
        }
        Arc::new(IoPool {
            tx_workers,
            handles,
            next_worker: AtomicUsize::new(0),
        })
    }

    /// Register a fresh ring with the pool. Returns the voice-facing
    /// handle. The pool will refill the ring in the background.
    ///
    /// `head` is the resident head buffer for THIS voice's region
    /// (per-channel). `head_start` is the file frame where the head
    /// buffer begins (usually 0, or the region's `offset=` value).
    /// The ring is initialized to start filling at `head_start +
    /// head.len()` so it picks up seamlessly past the head.
    ///
    /// Round-robin assigns the registration to one of the worker
    /// threads. The ring stays bound to that worker for its lifetime
    /// (no work-stealing) — simpler, no shared-state locking.
    pub fn register(
        &self,
        source: Arc<StreamedSampleSource>,
        channel: usize,
        head: Arc<[i16]>,
        head_start: usize,
    ) -> VoiceStream {
        let head_end = head_start + head.len();
        let ring = Arc::new(StreamRing::new_at(head_end));
        let reg = RegisteredRing {
            source: source.clone(),
            channel,
            ring: Arc::downgrade(&ring),
            flac_state: None,
        };
        let idx = self.next_worker.fetch_add(1, Ordering::Relaxed)
            % self.tx_workers.len();
        // Try to register; if the channel is full or shut down, the
        // voice still works — head buffer covers the spawn window,
        // and the ring will just stay empty (everything past head = silence).
        let _ = self.tx_workers[idx].try_send(PoolCommand::Register(reg));
        VoiceStream { source, channel, ring, head, head_start }
    }
}

impl Drop for IoPool {
    fn drop(&mut self) {
        for tx in &self.tx_workers {
            let _ = tx.send(PoolCommand::Shutdown);
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn pool_thread(rx: Receiver<PoolCommand>) {
    let mut working_set: Vec<RegisteredRing> = Vec::with_capacity(64);
    // Per-pass scratch buffers, reused across refills:
    //  - sep_buf: u16-decoded scratch for Separated-layout reads
    //  - inter_bytes: raw byte buffer for Interleaved-layout reads,
    //    sized for up to 8-channel frames at max chunk size
    let mut sep_buf: Vec<i16> = vec![0; REFILL_CHUNK_FRAMES];
    let mut inter_bytes: Vec<u8> = vec![0u8; REFILL_CHUNK_FRAMES * 16];
    let mut inter_decoded: Vec<i16> = vec![0i16; REFILL_CHUNK_FRAMES];

    loop {
        // Drain any pending commands (non-blocking).
        while let Ok(cmd) = rx.try_recv() {
            match cmd {
                PoolCommand::Register(r) => working_set.push(r),
                PoolCommand::Shutdown => return,
            }
        }

        // Scan working set; drop dead entries; refill those below
        // low-water.
        let mut did_work = false;
        let mut i = 0;
        while i < working_set.len() {
            let r = &mut working_set[i];
            let Some(ring) = r.ring.upgrade() else {
                // Voice dropped — remove from working set.
                working_set.swap_remove(i);
                continue;
            };

            // Already at EOF? Skip; ring will report 0 forever.
            if ring.eof.load(Ordering::Relaxed) {
                i += 1;
                continue;
            }

            let write_pos = ring.write_pos.load(Ordering::Relaxed);
            let consumed = ring.consumed_hint.load(Ordering::Acquire);
            let lead = write_pos.saturating_sub(consumed);

            if lead >= LOW_WATER_FRAMES {
                // Plenty of buffer ahead of the voice. Skip.
                i += 1;
                continue;
            }

            // Refill. Compute how many frames we can pull from disk.
            let total_frames = r.source.frames;
            let frames_remaining = total_frames.saturating_sub(write_pos);
            if frames_remaining == 0 {
                ring.set_eof();
                i += 1;
                continue;
            }

            let budget = ring.refill_budget().min(frames_remaining);
            if budget == 0 {
                i += 1;
                continue;
            }

            // FLAC streams write to the ring directly through their own
            // helper (per-ring decoder state, packet-driven copy). Other
            // layouts use the shared scratch + write_chunk plumbing
            // below.
            if matches!(r.source.layout, SampleLayout::Flac { .. }) {
                let (wrote, hit_eof) =
                    refill_flac_ring(r, &ring, write_pos, budget);
                if wrote > 0 {
                    did_work = true;
                }
                if hit_eof {
                    ring.set_eof();
                }
                i += 1;
                continue;
            }

            // Read `budget` frames of channel `r.channel` from the file
            // starting at frame `write_pos`. Layout determines the
            // byte-offset arithmetic and whether we stride.
            let read_result = match &r.source.layout {
                SampleLayout::Separated { byte_offset_per_channel } => {
                    let byte_offset = byte_offset_per_channel[r.channel]
                        + (write_pos as u64) * StreamedSampleSource::BYTES_PER_RING_FRAME;
                    let byte_len = budget * 2;
                    // SAFETY: sep_buf has capacity REFILL_CHUNK_FRAMES, budget <= REFILL_CHUNK_FRAMES.
                    let u8_buf = unsafe {
                        std::slice::from_raw_parts_mut(
                            sep_buf.as_mut_ptr() as *mut u8,
                            byte_len,
                        )
                    };
                    match r.source.file.read_at(u8_buf, byte_offset) {
                        Ok(n) if n == byte_len => Some((&sep_buf[..budget], false)),
                        Ok(n) => {
                            let frames_read = n / 2;
                            Some((&sep_buf[..frames_read], true))
                        }
                        Err(_) => None,
                    }
                }
                SampleLayout::Interleaved { data_byte_offset, frame_bytes } => {
                    let fb = *frame_bytes as usize;
                    let byte_offset = data_byte_offset
                        + (write_pos as u64) * (*frame_bytes as u64);
                    let byte_len = budget * fb;
                    // Grow scratch if needed (rare — only on first refill with
                    // unusually wide frames; bytes are reused after).
                    if inter_bytes.len() < byte_len {
                        inter_bytes.resize(byte_len, 0);
                    }
                    match r.source.file.read_at(&mut inter_bytes[..byte_len], byte_offset) {
                        Ok(n) => {
                            let frames_read = n / fb;
                            // Stride: extract this channel's i16 (LE) from
                            // each frame at `channel * 2` bytes.
                            let ch_byte_offset = r.channel * 2;
                            for f in 0..frames_read {
                                let lo = inter_bytes[f * fb + ch_byte_offset];
                                let hi = inter_bytes[f * fb + ch_byte_offset + 1];
                                inter_decoded[f] = i16::from_le_bytes([lo, hi]);
                            }
                            let short = n < byte_len;
                            Some((&inter_decoded[..frames_read], short))
                        }
                        Err(_) => None,
                    }
                }
                SampleLayout::Flac { .. } => {
                    // Handled by the early-return refill_flac_ring path above.
                    unreachable!()
                }
            };
            match read_result {
                Some((chunk, short)) => {
                    if !chunk.is_empty() {
                        ring.write_chunk(chunk);
                        did_work = true;
                    }
                    if short {
                        ring.set_eof();
                    }
                }
                None => ring.set_eof(),
            }
            i += 1;
        }

        // If we did nothing this pass, sleep briefly. Otherwise loop
        // immediately to keep up with hungry voices.
        if !did_work && working_set.is_empty() {
            // No rings registered — wait for one. Blocks until command
            // arrives or shutdown.
            match rx.recv() {
                Ok(PoolCommand::Register(r)) => working_set.push(r),
                Ok(PoolCommand::Shutdown) | Err(_) => return,
            }
        } else if !did_work {
            // Some rings registered but none needed refill — short
            // sleep so we don't burn CPU. 2 ms is well below the
            // smallest realistic refill interval (~23 ms = one chunk
            // played out at 1x).
            thread::sleep(Duration::from_millis(2));
        }
    }
}

/// MOVE FORK / 2026-05-17 (day 2): FLAC ring refill. Lazily opens a
/// per-ring symphonia decoder on first call, seeks to `write_pos`, then
/// decodes packets and copies the requested channel into the ring.
/// Leftover decoded samples beyond `budget` stay in `pending` for the
/// next refill so packet boundaries don't force per-call decoder churn.
///
/// Returns (frames written, hit_eof).
fn refill_flac_ring(
    r: &mut RegisteredRing,
    ring: &StreamRing,
    write_pos: usize,
    budget: usize,
) -> (usize, bool) {
    // Pull the FLAC path out of the source's layout. Caller guarantees
    // layout is Flac; anything else is a programmer error.
    let path = match &r.source.layout {
        SampleLayout::Flac { path } => path.clone(),
        _ => return (0, true),
    };
    let channel = r.channel;

    // Lazy init: open decoder + seek on first refill. Cost (~1-5 ms per
    // ring) is paid once on the I/O thread, never on the audio thread.
    if r.flac_state.is_none() {
        let Some((mut format, decoder, track_id)) = open_flac_decoder(&path) else {
            return (0, true);
        };
        let seeked = match format.seek(
            SeekMode::Accurate,
            SeekTo::TimeStamp { ts: write_pos as u64, track_id },
        ) {
            Ok(s) => s,
            Err(_) => return (0, true),
        };
        r.flac_state = Some(FlacRingState {
            format,
            decoder,
            track_id,
            decoder_cursor: seeked.actual_ts as usize,
            pending: Vec::new(),
            pending_start: write_pos,
        });
    }
    let state = r.flac_state.as_mut().unwrap();

    let mut hit_eof = false;
    while state.pending.len() < budget {
        let packet = match state.format.next_packet() {
            Ok(p) => p,
            Err(_) => { hit_eof = true; break; }
        };
        if packet.track_id() != state.track_id {
            continue;
        }
        // DecodeError leaves the packet's intrinsic length unknown, which
        // would break decoder_cursor accounting for subsequent packets.
        // Treat any decode failure as a hard stop (rare; audible only as
        // truncated tail, never as a drift glitch).
        let audio = match state.decoder.decode(&packet) {
            Ok(a) => a,
            Err(_) => { hit_eof = true; break; }
        };
        let pf = packet_frame_count(&audio);
        let packet_start = state.decoder_cursor;
        let packet_end = packet_start + pf;
        state.decoder_cursor = packet_end;

        // Where we want the next pending sample to land in absolute file
        // frames. After init this is `write_pos` (since pending is empty
        // and pending_start = write_pos); grows by packet contributions.
        let desired = state.pending_start + state.pending.len();
        if desired >= packet_end {
            // Packet is entirely before what we want — skip leading
            // samples that fell out of an Accurate seek's bracket.
            continue;
        }
        let local_start = desired.saturating_sub(packet_start);
        extract_packet_channel_i16(
            &audio, channel, local_start, pf, &mut state.pending,
        );
    }

    let take = budget.min(state.pending.len());
    if take > 0 {
        ring.write_chunk(&state.pending[..take]);
        state.pending.drain(..take);
        state.pending_start += take;
    }
    (take, hit_eof)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tempfile_path(prefix: &str) -> std::path::PathBuf {
        let pid = std::process::id();
        let nonce: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        std::env::temp_dir().join(format!("{}.{}.{}.dat", prefix, pid, nonce))
    }

    /// Test value at (channel, frame). Wraps to fit in i15 so the cast
    /// to i16 is lossless and `v as i32 == fixture_value(c, f) as i32`
    /// for any (c, f) the test reads.
    fn fixture_value(c: usize, f: usize) -> i16 {
        ((c as i32 * 10000 + f as i32) & 0x7fff) as i16
    }

    /// Build a StreamedSampleSource backed by a temp file with the
    /// deterministic fixture content. HEAD_FRAMES now exceeds i16::MAX,
    /// so we wrap the value modulo 0x7fff to keep cast comparisons
    /// straightforward.
    fn make_source(frames: usize, n_chans: usize) -> Arc<StreamedSampleSource> {
        let tmp = tempfile_path("xsynth-stream-test");
        {
            let mut file = File::create(&tmp).unwrap();
            for c in 0..n_chans {
                let mut buf: Vec<u8> = Vec::with_capacity(frames * 2);
                for f in 0..frames {
                    let v = fixture_value(c, f);
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                file.write_all(&buf).unwrap();
            }
        }
        let byte_offsets: Vec<u64> =
            (0..n_chans).map(|c| (c * frames * 2) as u64).collect();

        let f = File::open(&tmp).unwrap();
        Arc::new(StreamedSampleSource {
            file: Arc::new(f),
            layout: SampleLayout::Separated {
                byte_offset_per_channel: byte_offsets,
            },
            frames,
            n_chans,
            src_rate: 44100,
            data_rate: 44100,
        })
    }

    /// Build a per-channel head buffer (one Arc<[i16]> per channel) of
    /// HEAD_FRAMES frames starting at file frame 0. Mirrors the
    /// historical "head at offset 0" pattern used by tests.
    fn head_at_zero(source: &StreamedSampleSource) -> Vec<Arc<[i16]>> {
        let head_len = HEAD_FRAMES.min(source.frames);
        (0..source.n_chans)
            .map(|c| {
                let h: Vec<i16> = (0..head_len).map(|f| fixture_value(c, f)).collect();
                Arc::from(h.into_boxed_slice())
            })
            .collect()
    }

    #[test]
    fn ring_underrun_before_fill() {
        let ring = StreamRing::new();
        // Position past head boundary but not yet filled — underrun.
        assert!(ring.get(HEAD_FRAMES).is_none());
    }

    #[test]
    fn ring_fill_and_read() {
        let ring = StreamRing::new();
        let chunk: Vec<i16> = (0..256).map(|i| i as i16).collect();
        ring.write_chunk(&chunk);
        for i in 0..256 {
            assert_eq!(
                ring.get(HEAD_FRAMES + i),
                Some(i as i16),
                "read at pos {}",
                HEAD_FRAMES + i
            );
        }
        assert!(ring.get(HEAD_FRAMES + 256).is_none());
    }

    #[test]
    fn ring_wrap_around_capacity() {
        let ring = StreamRing::new();
        // Fill the ring most of the way (chunk fits in i16 range since
        // RING_FRAMES = 8192 < 32767).
        let chunk1: Vec<i16> = (0..(RING_FRAMES - 100)).map(|i| i as i16).collect();
        ring.write_chunk(&chunk1);
        // Consume far enough that another chunk can fit.
        ring.mark_consumed(HEAD_FRAMES + RING_FRAMES - 200);
        // Write another chunk; positions extend past the ring's "first
        // wrap" (the data array index wraps but absolute frame indices
        // keep counting).
        let chunk2: Vec<i16> = (0..200).map(|i| 9000 + i as i16).collect();
        ring.write_chunk(&chunk2);
        let start = HEAD_FRAMES + (RING_FRAMES - 100);
        for i in 0..200 {
            assert_eq!(
                ring.get(start + i),
                Some(9000 + i as i16),
                "read at pos {}",
                start + i
            );
        }
    }

    #[test]
    fn ring_eof_returns_zero() {
        let ring = StreamRing::new();
        ring.set_eof();
        // Past write_pos but EOF set → Some(0), not None.
        assert_eq!(ring.get(HEAD_FRAMES + 100), Some(0));
    }

    #[test]
    fn voice_stream_head_read() {
        let source = make_source(HEAD_FRAMES + 8, 2);
        let heads = head_at_zero(&source);
        let ring = Arc::new(StreamRing::new_at(HEAD_FRAMES));
        let vs = VoiceStream {
            source: source.clone(),
            channel: 1,
            ring,
            head: heads[1].clone(),
            head_start: 0,
        };
        assert_eq!(vs.get(0), fixture_value(1, 0));
        assert_eq!(vs.get(1), fixture_value(1, 1));
        assert_eq!(vs.get(HEAD_FRAMES - 1), fixture_value(1, HEAD_FRAMES - 1));
        // Past head, ring is empty → underrun → 0.
        assert_eq!(vs.get(HEAD_FRAMES), 0);
    }

    #[cfg(unix)]
    #[test]
    fn io_pool_refills_ring() {
        let frames = HEAD_FRAMES + 4096;
        let source = make_source(frames, 1);
        let heads = head_at_zero(&source);
        let pool = IoPool::new();
        let vs = pool.register(source.clone(), 0, heads[0].clone(), 0);

        // Wait briefly for the pool thread to fill the ring.
        let deadline = std::time::Instant::now() + Duration::from_millis(500);
        let mut filled = false;
        while std::time::Instant::now() < deadline {
            if vs.ring.write_pos.load(Ordering::Acquire) >= HEAD_FRAMES + 1024 {
                filled = true;
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert!(filled, "pool did not refill ring within 500 ms");

        // Read from the ring; values should match the fixture.
        for pos in HEAD_FRAMES..(HEAD_FRAMES + 512) {
            let v = vs.get(pos);
            assert_eq!(v, fixture_value(0, pos), "voice read at pos {}", pos);
            vs.ring.mark_consumed(pos);
        }
    }

    /// MOVE FORK / 2026-05-17 (day 2): build a synthetic FLAC file
    /// large enough to exercise BOTH the head buffer (44100 frames) and
    /// the post-head ring refill path. Generated via the `flac` CLI
    /// (homebrew on Mac / apt on Linux) from a raw PCM stream. Skipped
    /// if `flac` isn't on PATH.
    fn build_test_flac() -> Option<(std::path::PathBuf, Vec<i16>)> {
        // 2.5 seconds of mono 44.1 kHz @ 16-bit = 110250 frames.
        let frames: usize = 110_250;
        let mut samples = Vec::with_capacity(frames);
        for f in 0..frames {
            // Deterministic content: sine + per-sample-distinct index
            // so off-by-one comparison failures are visible.
            let t = (f as f32 / 44100.0) * 2.0 * std::f32::consts::PI * 440.0;
            let v = (t.sin() * 0.5 * 32767.0) as i16;
            // XOR in the low bits of the frame index so consecutive
            // samples differ even when sin returns the same value.
            samples.push(v ^ ((f as i16) & 0x0F));
        }

        let wav = tempfile_path("xsynth-flac-test-wav");
        let flac = wav.with_extension("flac");
        // Write WAV header + data manually (canonical PCM/16-bit/mono).
        {
            let mut f = File::create(&wav).ok()?;
            let data_bytes: u32 = (frames * 2) as u32;
            let header: Vec<u8> = {
                let mut h = Vec::with_capacity(44);
                h.extend_from_slice(b"RIFF");
                h.extend_from_slice(&(36 + data_bytes).to_le_bytes());
                h.extend_from_slice(b"WAVE");
                h.extend_from_slice(b"fmt ");
                h.extend_from_slice(&16u32.to_le_bytes());
                h.extend_from_slice(&1u16.to_le_bytes());   // PCM
                h.extend_from_slice(&1u16.to_le_bytes());   // mono
                h.extend_from_slice(&44100u32.to_le_bytes());
                h.extend_from_slice(&(44100u32 * 2).to_le_bytes()); // byte rate
                h.extend_from_slice(&2u16.to_le_bytes());   // block align
                h.extend_from_slice(&16u16.to_le_bytes());  // bits per sample
                h.extend_from_slice(b"data");
                h.extend_from_slice(&data_bytes.to_le_bytes());
                h
            };
            f.write_all(&header).ok()?;
            let bytes = unsafe {
                std::slice::from_raw_parts(samples.as_ptr() as *const u8, frames * 2)
            };
            f.write_all(bytes).ok()?;
        }

        // Invoke `flac` to compress.
        let status = std::process::Command::new("flac")
            .arg("--silent")
            .arg("--force")
            .arg("--no-padding")
            .arg("-o").arg(&flac)
            .arg(&wav)
            .status()
            .ok()?;
        let _ = std::fs::remove_file(&wav);
        if !status.success() {
            return None;
        }
        Some((flac, samples))
    }

    /// MOVE FORK / 2026-05-17 (day 2): smoke test the FLAC streaming
    /// path end-to-end. Generates a synthetic 110k-frame FLAC at load
    /// time, then verifies:
    ///   1. read_head_at(0) returns HEAD_FRAMES samples matching source
    ///   2. IoPool refills the ring past the head boundary
    ///   3. post-head ring samples match the source
    ///
    /// Skipped if the `flac` CLI is not installed on the test host.
    #[cfg(unix)]
    #[test]
    fn flac_stream_smoke() {
        let Some((path, reference)) = build_test_flac() else {
            eprintln!("flac_stream_smoke: flac CLI unavailable, skipping");
            return;
        };

        assert!(reference.len() > HEAD_FRAMES,
            "test fixture must exceed HEAD_FRAMES to exercise ring refill: {} <= {}",
            reference.len(), HEAD_FRAMES);

        let (sr, n_chans, n_frames) = {
            let (format, _decoder, _id) =
                open_flac_decoder(&path).expect("probe");
            let track = format.default_track().unwrap();
            (
                track.codec_params.sample_rate.unwrap(),
                track.codec_params.channels.unwrap().count(),
                track.codec_params.n_frames.unwrap() as usize,
            )
        };
        assert_eq!(sr, 44100);
        assert_eq!(n_chans, 1);
        assert_eq!(n_frames, reference.len());

        let file = std::fs::File::open(&path).unwrap();
        let source = Arc::new(StreamedSampleSource {
            file: Arc::new(file),
            layout: SampleLayout::Flac { path: path.clone() },
            frames: n_frames,
            n_chans,
            src_rate: sr,
            data_rate: sr,
        });

        // Head read at frame 0: must match reference for [0, HEAD_FRAMES).
        let heads = source.read_head_at(0).expect("read_head_at");
        assert_eq!(heads.len(), n_chans);
        assert_eq!(heads[0].len(), HEAD_FRAMES);
        for i in 0..HEAD_FRAMES {
            assert_eq!(
                heads[0][i], reference[i],
                "head sample {} mismatch", i,
            );
        }

        // Register the ring and let the pool refill past HEAD_FRAMES.
        let pool = IoPool::new();
        let vs = pool.register(source.clone(), 0, heads[0].clone(), 0);
        let head_end = HEAD_FRAMES;
        let want_frames = (head_end + 8192).min(n_frames);
        assert!(want_frames > head_end,
            "test fixture too small to exercise post-head ring");

        let deadline = std::time::Instant::now() + Duration::from_millis(3000);
        while std::time::Instant::now() < deadline {
            if vs.ring.write_pos.load(Ordering::Acquire) >= want_frames {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        let final_write = vs.ring.write_pos.load(Ordering::Acquire);
        assert!(final_write >= want_frames,
            "ring did not fill to {} within 3s (got {})", want_frames, final_write);

        // Post-head ring samples must match reference.
        for pos in head_end..want_frames {
            let got = vs.get(pos);
            let want = reference[pos];
            assert_eq!(got, want,
                "stream sample {} mismatch: got {} want {}", pos, got, want);
            vs.ring.mark_consumed(pos);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn read_head_at_offset() {
        // Verify that read_head_at returns the correct frames when
        // called with a non-zero offset — exactly the path that fixes
        // the Legacy Knight click bug.
        let frames = HEAD_FRAMES * 4;
        let source = make_source(frames, 2);
        let offset = HEAD_FRAMES * 2; // 88200 frames into the file
        let heads = source.read_head_at(offset).expect("read_head_at");
        assert_eq!(heads.len(), 2);
        // Both channels' heads should match the fixture content at
        // file frame `offset + i`, for i in 0..HEAD_FRAMES.
        for c in 0..2 {
            for i in 0..HEAD_FRAMES.min(frames - offset) {
                assert_eq!(
                    heads[c][i],
                    fixture_value(c, offset + i),
                    "head[{}][{}] vs fixture", c, i
                );
            }
        }
    }
}
