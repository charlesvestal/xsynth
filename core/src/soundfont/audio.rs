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

type ProcessedSample = (Arc<[Arc<SampleStorage>]>, u32);

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
