use std::{fs::File, io, io::Write, path::PathBuf, sync::Arc};

use symphonia::core::formats::FormatOptions;
use symphonia::core::{audio::AudioBuffer, conv::IntoSample, probe::Hint, sample::Sample};
use symphonia::core::{audio::AudioBufferRef, meta::MetadataOptions};
use symphonia::core::{audio::Signal, io::MediaSourceStream};
use symphonia::core::{codecs::DecoderOptions, errors::Error};

use crate::{AudioStreamParams, ChannelCount};
use thiserror::Error;
use xsynth_soundfonts::resample::resample_vecs;

/* MOVE FORK: per-sample mmap-backed cache. Layout (LE):
 *   magic    [u8; 4]  = "X44Y"  ("X44" = decoded f32 → now Y for i16 v2)
 *   version  u32      = 2
 *   src_rate u32
 *   tgt_rate u32
 *   tgt_chs  u32  (1 mono, 2 stereo)
 *   n_chans  u32
 *   frames   u32  (per channel)
 *   data     [i16; n_chans * frames]  (channel-contiguous)
 *
 * Voices read each i16 from the mmap'd region; OS pages out unused
 * samples under memory pressure. */
const CACHE_MAGIC: u32 = u32::from_le_bytes(*b"X44Y");
const CACHE_VERSION: u32 = 2;
const CACHE_HEADER_WORDS: usize = 7;
const CACHE_HEADER_BYTES: usize = CACHE_HEADER_WORDS * 4;

fn cache_path_for(source: &PathBuf) -> PathBuf {
    let mut p = source.clone();
    let mut name = p.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".x44c");
    p.set_file_name(name);
    p
}

fn try_read_u32(bytes: &[u8], i: usize) -> Option<u32> {
    let start = i * 4;
    if start + 4 > bytes.len() { return None; }
    Some(u32::from_le_bytes([bytes[start], bytes[start+1], bytes[start+2], bytes[start+3]]))
}

fn try_mmap_cache(
    source: &PathBuf,
    target_rate: u32,
    target_chans: ChannelCount,
) -> Option<(Arc<[Arc<SampleStorage>]>, u32)> {
    let cp = cache_path_for(source);
    let src_meta = std::fs::metadata(source).ok()?;
    let cache_meta = std::fs::metadata(&cp).ok()?;
    let src_mtime = src_meta.modified().ok()?;
    let cache_mtime = cache_meta.modified().ok()?;
    if cache_mtime < src_mtime { return None; }

    let file = File::open(&cp).ok()?;
    let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };
    if mmap.len() < CACHE_HEADER_BYTES { return None; }

    if try_read_u32(&mmap, 0)? != CACHE_MAGIC { return None; }
    if try_read_u32(&mmap, 1)? != CACHE_VERSION { return None; }
    let src_rate = try_read_u32(&mmap, 2)?;
    let cached_tgt_rate = try_read_u32(&mmap, 3)?;
    let cached_tgt_chs = try_read_u32(&mmap, 4)?;
    let n_chans = try_read_u32(&mmap, 5)? as usize;
    let frames = try_read_u32(&mmap, 6)? as usize;

    if cached_tgt_rate != target_rate { return None; }
    if cached_tgt_chs != target_chans.count() as u32 { return None; }
    if n_chans == 0 || frames == 0 || n_chans > 8 { return None; }
    let bytes_per_chan = frames * 2;
    if mmap.len() < CACHE_HEADER_BYTES + n_chans * bytes_per_chan { return None; }

    // MOVE FORK / 2026-05-16: copy the cache into a heap-backed Vec<i16>
    // and drop the mmap. The OS could evict mmap pages under memory
    // pressure, faulting them back in on the audio thread (~100 ms+
    // spikes seen on Move with the rhodes preset). Heap allocations are
    // never evicted, so audio rendering is fault-free at the cost of
    // ~src_size bytes of RAM that don't get to ride disk cache.
    let channels: Vec<Arc<SampleStorage>> = (0..n_chans).map(|c| {
        let start = CACHE_HEADER_BYTES + c * bytes_per_chan;
        let end   = start + bytes_per_chan;
        // SAFETY: cache header validated above so we're inside `mmap`.
        // i16 is POD; bytes here are the raw little-endian sample data.
        let src: &[u8] = &mmap[start..end];
        let mut buf: Vec<i16> = Vec::with_capacity(frames);
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr() as *const i16,
                buf.as_mut_ptr(),
                frames,
            );
            buf.set_len(frames);
        }
        Arc::new(SampleStorage::from_heap(Arc::from(buf.into_boxed_slice())))
    }).collect();
    // mmap drops here — pages can be reclaimed by the OS immediately.
    drop(mmap);
    Some((channels.into(), src_rate))
}

fn try_write_cache(
    source: &PathBuf,
    target_rate: u32,
    target_chans: ChannelCount,
    src_rate: u32,
    channels: &[Arc<[i16]>],
) {
    let cp = cache_path_for(source);
    let mut tmp = cp.clone();
    let mut tmpname = tmp.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    tmpname.push(".tmp");
    tmp.set_file_name(tmpname);
    let frames = channels.first().map(|c| c.len() as u32).unwrap_or(0);
    let n_chans = channels.len() as u32;
    if frames == 0 || n_chans == 0 { return; }

    let Ok(f) = File::create(&tmp) else { return; };
    let mut w = std::io::BufWriter::with_capacity(64 * 1024, f);
    let hdr: [u32; CACHE_HEADER_WORDS] = [
        CACHE_MAGIC, CACHE_VERSION, src_rate, target_rate,
        target_chans.count() as u32, n_chans, frames,
    ];
    let mut hdr_bytes = [0u8; CACHE_HEADER_BYTES];
    for (i, v) in hdr.iter().enumerate() {
        hdr_bytes[i*4..i*4+4].copy_from_slice(&v.to_le_bytes());
    }
    if w.write_all(&hdr_bytes).is_err() { return; }
    for ch in channels.iter() {
        if ch.len() as u32 != frames { return; }
        // SAFETY: i16 is POD. Bulk write the channel data as bytes.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(ch.as_ptr() as *const u8, ch.len() * 2)
        };
        if w.write_all(bytes).is_err() { return; }
    }
    if w.into_inner().is_err() { return; }
    let _ = std::fs::rename(&tmp, &cp);
}

/// Errors that can be generated when loading an audio file.
#[derive(Debug, Error)]
pub enum AudioLoadError {
    #[error("IO Error")]
    IOError(#[from] io::Error),

    #[error("Audio decoding failed for {0}")]
    AudioDecodingFailed(PathBuf, Error),

    #[error("Audio file {0} has an invalid channel count")]
    InvalidChannelCount(PathBuf),

    #[error("Audio file {0} has no tracks")]
    NoTracks(PathBuf),
}

use super::sample_storage::{MmapHolder, SampleStorage};
#[cfg(unix)]
use super::streaming::{StreamedSampleSource, HEAD_FRAMES};
#[cfg(unix)]
use std::os::unix::fs::FileExt;

type ProcessedSample = (Arc<[Arc<SampleStorage>]>, u32);

#[cfg(unix)]
type StreamedSample = (Arc<StreamedSampleSource>, u32);

/// MOVE FORK / 2026-05-16: pre-bake the `.x44c` cache for a single audio
/// file. Returns Ok(()) when the cache exists and is current after this
/// call. Used by the Mac-side `prebake_cache` CLI to convert sample
/// libraries (Salamander Grand Piano etc.) into Move-streamable form
/// without doing the slow FLAC decode on the device.
///
/// Triggers `load_audio_file` (full decode + cache write) when needed,
/// discards the heap result. Fast path when cache is already current.
pub fn prebake_audio_cache(
    path: &std::path::Path,
    stream_params: AudioStreamParams,
) -> Result<(), AudioLoadError> {
    let pb = path.to_path_buf();
    let _ = load_audio_file(&pb, stream_params)?;
    Ok(())
}

pub(super) fn load_audio_file(
    path: &PathBuf,
    stream_params: AudioStreamParams,
) -> Result<ProcessedSample, AudioLoadError> {
    let new_sample_rate = stream_params.sample_rate as f32;

    // Cache hit: skip decode entirely; samples are mmap'd, RAM cost is
    // bounded to actively-played pages.
    if let Some(cached) = try_mmap_cache(path, stream_params.sample_rate, stream_params.channels) {
        return Ok(cached);
    }

    let extension = path.extension().and_then(|ext| ext.to_str());

    let file = Box::new(File::open(path)?);

    // Create the media source stream using the boxed media source from above.
    let mss = MediaSourceStream::new(file, Default::default());

    // Create a hint to help the format registry guess what format reader is appropriate.
    let mut hint = Hint::new();
    if let Some(extension) = extension {
        hint.with_extension(extension);
    }

    // Use the default options when reading and decoding.
    let format_opts: FormatOptions = Default::default();
    let metadata_opts: MetadataOptions = Default::default();
    let decoder_opts: DecoderOptions = Default::default();

    // Probe the media source stream for a format.
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|x| AudioLoadError::AudioDecodingFailed(path.clone(), x))?;

    // Get the format reader yielded by the probe operation.
    let mut format = probed.format;

    // Get the default track.
    let track = format
        .default_track()
        .ok_or_else(|| AudioLoadError::NoTracks(path.clone()))?;

    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channel_count = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|x| AudioLoadError::AudioDecodingFailed(path.clone(), x))?;

    // Store the track identifier, we'll use it to filter packets.
    let track_id = track.id;

    // Builder for the parsed audio buffers
    let mut builder = BuilderVecs::new(channel_count);

    loop {
        // Get the next packet from the format reader.
        let packet = match format.next_packet() {
            Err(symphonia::core::errors::Error::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                // Audio source ended. Currently the lib has no cleaner way of detecting this.
                break;
            }
            Err(error) => return Err(AudioLoadError::AudioDecodingFailed(path.clone(), error)),
            Ok(packet) => packet,
        };

        // If the packet does not belong to the selected track, skip it.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples, ignoring any decode errors.
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                builder.push(audio_buf);
            }

            Err(Error::DecodeError(_)) => (),
            Err(e) => return Err(AudioLoadError::AudioDecodingFailed(path.clone(), e)),
        }
    }

    let built_i16 = builder.finish(sample_rate as f32, new_sample_rate, stream_params.channels);

    // Write cache so subsequent loads are mmap-fast. Best effort; failure
    // is harmless (we just decode again next time).
    let chans_vec: Vec<Arc<[i16]>> = built_i16.iter().cloned().collect();
    try_write_cache(
        path,
        stream_params.sample_rate,
        stream_params.channels,
        sample_rate,
        &chans_vec,
    );

    // Try to mmap the freshly-written cache so even the first load benefits
    // from page-eviction-under-pressure. Fall back to heap-wrapped storage
    // if mmap fails (e.g. write failed silently).
    if let Some(mapped) = try_mmap_cache(path, stream_params.sample_rate, stream_params.channels) {
        return Ok(mapped);
    }
    let heap: Arc<[Arc<SampleStorage>]> = built_i16
        .iter()
        .map(|chan| Arc::new(SampleStorage::from_heap(chan.clone())))
        .collect();
    Ok((heap, sample_rate))
}

/// MOVE FORK / 2026-05-16: streamed equivalent of `load_audio_file`.
/// Returns an open file handle + per-channel byte offsets + a small
/// always-resident head buffer. The voice spawner uses this to construct
/// per-voice ring buffers that an IoPool thread fills in the background.
///
/// On cache miss, falls back to `load_audio_file` (which decodes and
/// writes the .x44c cache) and then re-opens for streaming. The heap
/// data returned by the fallback path is dropped — wasteful on first-
/// load but only one-time. Subsequent loads hit the cache directly.
#[cfg(unix)]
pub(super) fn load_audio_file_streamed(
    path: &PathBuf,
    stream_params: AudioStreamParams,
) -> Result<StreamedSample, AudioLoadError> {
    // MOVE FORK / 2026-05-16: direct WAV streaming. For canonical
    // 16-bit PCM WAVs at the target sample rate, skip the entire
    // decode + .x44c cache pass — stream straight from the .wav file.
    // Wins: zero prebake, instant first-load, no cache disk space.
    if let Some(s) =
        try_open_wav_streamed(path, stream_params.sample_rate, stream_params.channels)
    {
        return Ok(s);
    }
    if let Some(s) =
        try_open_streamed_cache(path, stream_params.sample_rate, stream_params.channels)
    {
        return Ok(s);
    }

    // Cache is missing or stale. Build it via the full decode path,
    // then drop the heap result and re-open as streamed.
    let _heap = load_audio_file(path, stream_params)?;
    drop(_heap);

    if let Some(s) =
        try_open_streamed_cache(path, stream_params.sample_rate, stream_params.channels)
    {
        return Ok(s);
    }

    // Should not happen: load_audio_file should have written the cache
    // successfully. If we get here, the cache write failed silently
    // (disk full, permissions, etc.) — surface as IO error.
    Err(AudioLoadError::IOError(io::Error::new(
        io::ErrorKind::Other,
        "streamed load: cache file not present after decode pass",
    )))
}

/// MOVE FORK / 2026-05-16: open an existing .x44c cache for streamed
/// access. Reads only the header + head buffer (first HEAD_FRAMES of
/// each channel); the rest of the sample data stays on disk, paged in
/// by the IoPool's pread calls as voices play.
///
/// Returns None if cache is missing, stale, or has mismatched params.
#[cfg(unix)]
fn try_open_streamed_cache(
    source: &PathBuf,
    target_rate: u32,
    target_chans: ChannelCount,
) -> Option<StreamedSample> {
    let cp = cache_path_for(source);
    let src_meta = std::fs::metadata(source).ok()?;
    let cache_meta = std::fs::metadata(&cp).ok()?;
    let src_mtime = src_meta.modified().ok()?;
    let cache_mtime = cache_meta.modified().ok()?;
    if cache_mtime < src_mtime { return None; }

    let file = File::open(&cp).ok()?;
    let mut hdr_bytes = [0u8; CACHE_HEADER_BYTES];
    if file.read_at(&mut hdr_bytes, 0).ok()? < CACHE_HEADER_BYTES {
        return None;
    }
    let read_u32 = |i: usize| -> u32 {
        u32::from_le_bytes([
            hdr_bytes[i * 4], hdr_bytes[i * 4 + 1],
            hdr_bytes[i * 4 + 2], hdr_bytes[i * 4 + 3],
        ])
    };
    if read_u32(0) != CACHE_MAGIC { return None; }
    if read_u32(1) != CACHE_VERSION { return None; }
    let src_rate = read_u32(2);
    let cached_tgt_rate = read_u32(3);
    let cached_tgt_chs = read_u32(4);
    let n_chans = read_u32(5) as usize;
    let frames = read_u32(6) as usize;

    if cached_tgt_rate != target_rate { return None; }
    if cached_tgt_chs != target_chans.count() as u32 { return None; }
    if n_chans == 0 || frames == 0 || n_chans > 8 { return None; }

    let bytes_per_chan = frames * 2;
    let expected_size = CACHE_HEADER_BYTES + n_chans * bytes_per_chan;
    if (cache_meta.len() as usize) < expected_size { return None; }

    // Per-channel byte offsets within the file. Channels are stored
    // back-to-back: [hdr][ch0 i16 data][ch1 i16 data]...
    let byte_offset_per_channel: Vec<u64> = (0..n_chans)
        .map(|c| (CACHE_HEADER_BYTES + c * bytes_per_chan) as u64)
        .collect();

    // MOVE FORK / 2026-05-16: head buffers are no longer per-sample;
    // they're per-(sample, region.offset). Caller (new_sfz_inner) loads
    // heads via source.read_head_at(offset) after construction.
    let source = Arc::new(StreamedSampleSource {
        file: Arc::new(file),
        layout: super::streaming::SampleLayout::Separated {
            byte_offset_per_channel,
        },
        frames,
        n_chans,
        src_rate,
    });
    Some((source, src_rate))
}

/// MOVE FORK / 2026-05-16: open a canonical 16-bit PCM WAV file as a
/// streamed source directly, no `.x44c` cache. Eligible iff the WAV is
/// exactly at the target sample rate, 16-bit, mono or stereo PCM. Any
/// non-match (different bit depth, sample rate, compressed format,
/// odd extension, non-canonical chunks) returns `None` and the caller
/// falls back to the cache + decode path.
///
/// Skips the entire decode + cache-write pass for 44/16 WAV libraries
/// like Salamander Grand Piano V3 (44.1 kHz / 16-bit variant): preset
/// load goes straight from filesystem to streamable handle, no
/// preprocessing, no prebake.
///
/// Cache files (.x44c) for the same path are intentionally ignored —
/// when the WAV already IS the streaming format, the cache is dead
/// weight.
#[cfg(unix)]
fn try_open_wav_streamed(
    source: &PathBuf,
    target_rate: u32,
    target_chans: ChannelCount,
) -> Option<StreamedSample> {
    // Only consider .wav extensions; .flac etc. go through symphonia.
    let ext = source.extension().and_then(|e| e.to_str()).map(str::to_lowercase);
    if ext.as_deref() != Some("wav") {
        return None;
    }

    let file = File::open(source).ok()?;
    // Read enough of the header to validate format + find the `data`
    // chunk. 12 bytes RIFF header + a few chunks; canonical PCM WAVs
    // are typically <= 128 bytes of header.
    let mut hdr = [0u8; 256];
    let n = file.read_at(&mut hdr, 0).ok()?;
    if n < 44 { return None; }

    // RIFF / WAVE.
    if &hdr[0..4] != b"RIFF" { return None; }
    if &hdr[8..12] != b"WAVE" { return None; }

    // Scan chunks for `fmt ` and `data`. Each chunk: 4-byte tag,
    // 4-byte LE size, then payload. Start at byte 12 (after WAVE).
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format_tag, n_chans, sample_rate, bits_per_sample)
    let mut data_offset: Option<u64> = None;
    let mut data_byte_len: Option<u64> = None;

    while pos + 8 <= n {
        let tag = &hdr[pos..pos + 4];
        let size = u32::from_le_bytes([hdr[pos + 4], hdr[pos + 5], hdr[pos + 6], hdr[pos + 7]]) as u64;
        if tag == b"fmt " && pos + 8 + 16 <= n {
            let format_tag = u16::from_le_bytes([hdr[pos + 8], hdr[pos + 9]]);
            let n_chans_raw = u16::from_le_bytes([hdr[pos + 10], hdr[pos + 11]]);
            let sample_rate = u32::from_le_bytes([
                hdr[pos + 12], hdr[pos + 13], hdr[pos + 14], hdr[pos + 15],
            ]);
            let bits_per_sample = u16::from_le_bytes([hdr[pos + 22], hdr[pos + 23]]);
            fmt = Some((format_tag, n_chans_raw, sample_rate, bits_per_sample));
            pos += 8 + size as usize;
            continue;
        }
        if tag == b"data" {
            data_offset = Some((pos + 8) as u64);
            data_byte_len = Some(size);
            break;
        }
        pos += 8 + size as usize;
    }

    let (format_tag, wav_chans, sample_rate, bits_per_sample) = fmt?;
    let data_offset = data_offset?;
    let data_byte_len = data_byte_len?;

    // Strict eligibility checks. Anything fancy → fall back.
    if format_tag != 1 { return None; }              // 1 = PCM
    if bits_per_sample != 16 { return None; }        // Only 16-bit
    if sample_rate != target_rate { return None; }   // Must match output rate
    if wav_chans == 0 || wav_chans > 2 { return None; } // Only mono/stereo

    let frame_bytes = (wav_chans as u32) * 2;
    let frames = (data_byte_len / frame_bytes as u64) as usize;
    if frames == 0 { return None; }
    let _ = target_chans;

    // MOVE FORK / 2026-05-16: per-(sample, offset) heads are loaded by
    // the caller via source.read_head_at(offset). Source itself carries
    // only the file handle + layout metadata.
    let source = Arc::new(StreamedSampleSource {
        file: Arc::new(file),
        layout: super::streaming::SampleLayout::Interleaved {
            data_byte_offset: data_offset,
            frame_bytes,
        },
        frames,
        n_chans: wav_chans as usize,
        src_rate: sample_rate,
    });
    Some((source, sample_rate))
}

struct BuilderVecs {
    vecs: Vec<Vec<f32>>,
}

impl BuilderVecs {
    fn new(channels: usize) -> Self {
        let mut vecs = Vec::new();
        for _ in 0..channels {
            vecs.push(Vec::new());
        }

        Self { vecs }
    }

    fn push(&mut self, buffer: AudioBufferRef) {
        match buffer {
            AudioBufferRef::U8(buf) => self.push_buffer(&buf),
            AudioBufferRef::U16(buf) => self.push_buffer(&buf),
            AudioBufferRef::U24(buf) => self.push_buffer(&buf),
            AudioBufferRef::U32(buf) => self.push_buffer(&buf),
            AudioBufferRef::S8(buf) => self.push_buffer(&buf),
            AudioBufferRef::S16(buf) => self.push_buffer(&buf),
            AudioBufferRef::S24(buf) => self.push_buffer(&buf),
            AudioBufferRef::S32(buf) => self.push_buffer(&buf),
            AudioBufferRef::F32(buf) => self.push_buffer(&buf),
            AudioBufferRef::F64(buf) => self.push_buffer(&buf),
        }
    }

    fn push_buffer(&mut self, buffer: &AudioBuffer<impl Sample + IntoSample<f32>>) {
        let channels = buffer.spec().channels.count();

        for c in 0..channels {
            let channel = buffer.chan(c);
            self.vecs[c].reserve(channel.len());
            for &sample in channel.iter() {
                self.vecs[c].push(sample.into_sample());
            }
        }
    }

    fn finish(
        self,
        sample_rate: f32,
        new_sample_rate: f32,
        channels: ChannelCount,
    ) -> Arc<[Arc<[i16]>]> {
        let mut vecs = self.vecs;

        if channels == ChannelCount::Mono && vecs.len() >= 2 {
            let right = vecs.pop().unwrap_or_default();
            let left = vecs.pop().unwrap_or_default();

            let combined: Vec<f32> = left
                .iter()
                .zip(right.iter())
                .map(|(&l, &r)| (l + r) * 0.5)
                .collect();
            vecs.push(combined);
        }

        for chan in vecs.iter_mut() {
            chan.shrink_to_fit();
        }

        resample_vecs(vecs, sample_rate, new_sample_rate)
    }
}
