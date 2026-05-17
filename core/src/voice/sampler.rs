use std::{marker::PhantomData, sync::Arc};

use simdeez::prelude::*;

use crate::soundfont::LoopParams;
use crate::voice::{ReleaseType, VoiceControlData};

use super::{SIMDSampleMono, SIMDSampleStereo, SIMDVoiceGenerator, VoiceGeneratorBase};

mod linear;
pub use linear::*;

mod nearest;
pub use nearest::*;

// I believe some terminology reference is relevant for this one.
//
// BufferSampler: Something that grabs a sample based on an index
//
// SampleReader: Something that grabs the sample value at an arbitrary index,
// and implements sample start/end/looping
//
// SIMDSampleGrabber: Something that takes a SIMD array of float64 locations and
// returns a SIMD array of f32 interpolated sample values

// Base traits

pub trait BufferSampler: Send + Sync {
    fn get(&self, pos: usize) -> f32;
    fn length(&self) -> usize;
}

pub trait SIMDSampleGrabber<S: Simd>: Send + Sync {
    /// Indexes: the rounded index of the sample
    ///
    /// Fractional: The fractional part of the index, i.e. the 0-1 range decimal
    fn get(&mut self, indexes: S::Vi32, fractional: S::Vf32) -> S::Vf32;

    fn is_past_end(&self, pos: f64) -> bool;

    fn signal_release(&mut self);
}

// MOVE FORK: samples are stored as i16 internally to halve memory footprint
// vs xsynth's original f32. Storage is either heap or mmap'd cache; the
// sampler is opaque to the choice via SampleStorage::get(pos) -> i16. The
// per-read cost is one i16→f32 multiply, negligible compared to the
// interpolation + envelope + filter work that follows.

use crate::soundfont::SampleStorage;

pub struct I16BufferSampler(Arc<SampleStorage>);

const I16_TO_F32: f32 = 1.0 / 32768.0;

impl BufferSampler for I16BufferSampler {
    #[inline(always)]
    fn get(&self, pos: usize) -> f32 {
        let v = self.0.get(pos);
        (v as f32) * I16_TO_F32
    }

    fn length(&self) -> usize {
        self.0.len()
    }
}

// MOVE FORK / 2026-05-16: streamed sample backend. Each voice owns one
// StreamedBufferSampler per channel (i.e. left+right for stereo). The
// VoiceStream inside holds an Arc to the shared StreamedSampleSource
// (resident head buffer + file handle) and an Arc to its own ring buffer
// that an I/O pool thread fills in the background. The audio thread
// never blocks on disk: positions < HEAD_FRAMES read from the resident
// head; later positions read from the ring (underrun → silence).
#[cfg(unix)]
pub struct StreamedBufferSampler {
    stream: crate::soundfont::VoiceStream,
}

#[cfg(unix)]
impl StreamedBufferSampler {
    pub fn new(stream: crate::soundfont::VoiceStream) -> Self {
        StreamedBufferSampler { stream }
    }
}

#[cfg(unix)]
impl BufferSampler for StreamedBufferSampler {
    #[inline(always)]
    fn get(&self, pos: usize) -> f32 {
        let v = self.stream.get(pos);
        (v as f32) * I16_TO_F32
    }

    fn length(&self) -> usize {
        self.stream.length()
    }
}

pub enum BufferSamplers {
    I16(I16BufferSampler),
    #[cfg(unix)]
    Streamed(StreamedBufferSampler),
}

impl BufferSamplers {
    #[inline(always)]
    pub fn new_f32(sample: Arc<SampleStorage>) -> BufferSamplers {
        BufferSamplers::I16(I16BufferSampler(sample))
    }

    #[cfg(unix)]
    #[inline(always)]
    pub fn streamed(stream: crate::soundfont::VoiceStream) -> BufferSamplers {
        BufferSamplers::Streamed(StreamedBufferSampler::new(stream))
    }
}

impl BufferSampler for BufferSamplers {
    #[inline(always)]
    fn get(&self, pos: usize) -> f32 {
        match self {
            BufferSamplers::I16(sampler) => sampler.get(pos),
            #[cfg(unix)]
            BufferSamplers::Streamed(sampler) => sampler.get(pos),
        }
    }

    fn length(&self) -> usize {
        match self {
            BufferSamplers::I16(sampler) => sampler.length(),
            #[cfg(unix)]
            BufferSamplers::Streamed(sampler) => sampler.length(),
        }
    }
}

// Enum sampler reader

pub trait SampleReader: Send + Sync {
    fn get(&mut self, pos: usize) -> f32;
    fn is_past_end(&self, pos: usize) -> bool;
    fn signal_release(&mut self);
}

pub struct SampleReaderNoLoop<Sampler: BufferSampler> {
    buffer: Sampler,
    length: Option<usize>,
    offset: usize,
}

impl<Sampler: BufferSampler> SampleReaderNoLoop<Sampler> {
    pub fn new(buffer: Sampler, loop_params: LoopParams) -> Self {
        let stop = loop_params
            .stop
            .map(|stop| stop as usize)
            .unwrap_or_else(|| buffer.length());
        let length = Some(stop);
        Self {
            buffer,
            length,
            offset: loop_params.offset as usize,
        }
    }
}

/// MOVE FORK / 2026-05-17: end-edge fade-out length in frames.
/// Some sample libraries (e.g. StereoRhodes) end mid-decay at a non-
/// zero amplitude (~-30 dBFS); no_loop playback hits EOF and snaps to
/// silence, producing an audible click. Linearly fading the last
/// FADE_FRAMES samples to zero hides the discontinuity. 256 frames
/// at 44.1 kHz ≈ 5.8 ms — short enough to be inaudible as a fade,
/// long enough to suppress the click on samples ending around -20 dB.
const SAMPLE_EDGE_FADE_FRAMES: usize = 256;

impl<Sampler: BufferSampler> SampleReader for SampleReaderNoLoop<Sampler> {
    fn get(&mut self, pos: usize) -> f32 {
        let buf_pos = pos + self.offset;
        let s = self.buffer.get(buf_pos);
        // End-edge fade: when within the last SAMPLE_EDGE_FADE_FRAMES of
        // the sample, ramp linearly to zero. Past the end the buffer
        // returns 0 anyway, so the fade only affects in-range reads.
        if let Some(len) = self.length {
            let fade_start = len.saturating_sub(SAMPLE_EDGE_FADE_FRAMES);
            if buf_pos >= fade_start {
                let remaining = len.saturating_sub(buf_pos);
                let fade = remaining as f32 / SAMPLE_EDGE_FADE_FRAMES as f32;
                return s * fade.min(1.0);
            }
        }
        s
    }

    fn is_past_end(&self, pos: usize) -> bool {
        if let Some(len) = self.length {
            pos >= len.saturating_sub(self.offset)
        } else {
            false
        }
    }

    fn signal_release(&mut self) {}
}

pub struct SampleReaderLoop<Sampler: BufferSampler> {
    buffer: Sampler,
    offset: usize,
    loop_start: usize,
    loop_end: usize,
    /// MOVE FORK / Phase 8: SFZ `loop_crossfade` in audio frames. The
    /// last N samples before `loop_end` blend with the pre-`loop_start`
    /// equivalent so the loop boundary is seamless. Zero disables.
    crossfade: usize,
}

impl<Sampler: BufferSampler> SampleReaderLoop<Sampler> {
    pub fn new(buffer: Sampler, loop_params: LoopParams) -> Self {
        // Clamp crossfade to the loop region length so it never reads
        // outside the buffer.
        let region = loop_params.end.saturating_sub(loop_params.start) as usize;
        let crossfade = (loop_params.crossfade as usize).min(region);
        Self {
            buffer,
            offset: loop_params.offset as usize,
            loop_start: loop_params.start as usize,
            loop_end: loop_params.end as usize,
            crossfade,
        }
    }
}

impl<Sampler: BufferSampler> SampleReader for SampleReaderLoop<Sampler> {
    fn get(&mut self, pos: usize) -> f32 {
        let mut pos = pos + self.offset;
        let end = self.loop_end;
        let start = self.loop_start;

        if pos > end {
            pos = (pos - end - 1) % (end - start) + start;
        }

        // MOVE FORK / Phase 8: linear crossfade across the last N
        // frames of the loop. At `pos == end - xfade`, alpha = 0 (pure
        // late-loop sample); at `pos == end`, alpha = 1 (pure
        // pre-start sample). The pre-start sample at `pos - (end -
        // start)` is what the audio would naturally pick up just past
        // the loop wrap, so blending into it smooths the discontinuity.
        if self.crossfade > 0 && end > self.crossfade && pos > end - self.crossfade {
            let into_xfade = pos - (end - self.crossfade);
            let alpha = into_xfade as f32 / self.crossfade as f32;
            let region = end - start;
            let primary = self.buffer.get(pos);
            // Underflow guard: only blend when the pre-start sample
            // exists in the buffer.
            if pos > region {
                let pre = self.buffer.get(pos - region);
                return primary * (1.0 - alpha) + pre * alpha;
            }
            return primary;
        }
        self.buffer.get(pos)
    }

    fn is_past_end(&self, _pos: usize) -> bool {
        false
    }

    fn signal_release(&mut self) {}
}

pub struct SampleReaderLoopSustain<Sampler: BufferSampler> {
    buffer: Sampler,
    length: Option<usize>,
    offset: usize,
    loop_start: usize,
    loop_end: usize,
    /// MOVE FORK / Phase 8: see SampleReaderLoop.
    crossfade: usize,
    last: usize,
    is_released: bool,
}

impl<Sampler: BufferSampler> SampleReaderLoopSustain<Sampler> {
    pub fn new(buffer: Sampler, loop_params: LoopParams) -> Self {
        let stop = loop_params
            .stop
            .map(|stop| stop as usize)
            .unwrap_or_else(|| buffer.length());
        let length = Some(stop);
        let region = loop_params.end.saturating_sub(loop_params.start) as usize;
        let crossfade = (loop_params.crossfade as usize).min(region);
        Self {
            buffer,
            length,
            offset: loop_params.offset as usize,
            loop_start: loop_params.start as usize,
            loop_end: loop_params.end as usize,
            crossfade,
            last: 0,
            is_released: false,
        }
    }
}

impl<Sampler: BufferSampler> SampleReader for SampleReaderLoopSustain<Sampler> {
    fn get(&mut self, pos: usize) -> f32 {
        let mut pos = pos + self.offset;
        let end = self.loop_end;
        let start = self.loop_start;

        if !self.is_released {
            self.last = pos;
            if pos > end {
                pos = (pos - end - 1) % (end - start) + start;
            }
            // MOVE FORK / Phase 8: crossfade across loop boundary.
            // Skipped while released (the sample plays linearly past
            // the loop point so the natural decay is preserved).
            if self.crossfade > 0 && end > self.crossfade && pos > end - self.crossfade {
                let into_xfade = pos - (end - self.crossfade);
                let alpha = into_xfade as f32 / self.crossfade as f32;
                let region = end - start;
                let primary = self.buffer.get(pos);
                if pos > region {
                    let pre = self.buffer.get(pos - region);
                    return primary * (1.0 - alpha) + pre * alpha;
                }
                return primary;
            }
        } else {
            pos = pos - self.last + self.loop_end;
        }

        self.buffer.get(pos)
    }

    fn is_past_end(&self, pos: usize) -> bool {
        if let Some(len) = self.length {
            pos >= len.saturating_sub(self.last + self.offset)
        } else {
            false
        }
    }

    fn signal_release(&mut self) {
        self.is_released = true;
    }
}

// Sample grabbers enum

pub enum SIMDSampleGrabbers<S: Simd, Reader: SampleReader> {
    Nearest(SIMDNearestSampleGrabber<S, Reader>),
    Linear(SIMDLinearSampleGrabber<S, Reader>),
}

impl<S: Simd, Reader: SampleReader> SIMDSampleGrabbers<S, Reader> {
    pub fn nearest(reader: Reader) -> Self {
        SIMDSampleGrabbers::Nearest(SIMDNearestSampleGrabber::new(reader))
    }

    pub fn linear(reader: Reader) -> Self {
        SIMDSampleGrabbers::Linear(SIMDLinearSampleGrabber::new(reader))
    }
}

impl<S: Simd, Reader: SampleReader> SIMDSampleGrabber<S> for SIMDSampleGrabbers<S, Reader> {
    #[inline(always)]
    fn get(&mut self, indexes: S::Vi32, fractional: S::Vf32) -> S::Vf32 {
        match self {
            SIMDSampleGrabbers::Linear(grabber) => grabber.get(indexes, fractional),
            SIMDSampleGrabbers::Nearest(grabber) => grabber.get(indexes, fractional),
        }
    }

    #[inline(always)]
    fn is_past_end(&self, pos: f64) -> bool {
        match self {
            SIMDSampleGrabbers::Linear(grabber) => grabber.is_past_end(pos),
            SIMDSampleGrabbers::Nearest(grabber) => grabber.is_past_end(pos),
        }
    }

    #[inline(always)]
    fn signal_release(&mut self) {
        match self {
            SIMDSampleGrabbers::Linear(grabber) => grabber.signal_release(),
            SIMDSampleGrabbers::Nearest(grabber) => grabber.signal_release(),
        }
    }
}

// Sampler generator

pub struct SIMDMonoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    grabber: Grabber,

    pitch_gen: Pitch,

    time: f64,

    _s: PhantomData<S>,
}

impl<S, Pitch, Grabber> SIMDMonoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    pub fn new(grabber: Grabber, pitch_gen: Pitch) -> Self {
        SIMDMonoVoiceSampler {
            grabber,
            pitch_gen,
            time: 0.0,
            _s: PhantomData,
        }
    }

    fn increment_time(&mut self, by: f64) -> f64 {
        let time = self.time;
        self.time += by;
        time
    }
}

impl<S, Pitch, Grabber> VoiceGeneratorBase for SIMDMonoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    #[inline(always)]
    fn ended(&self) -> bool {
        self.grabber.is_past_end(self.time)
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.pitch_gen.signal_release(rel_type);
        self.grabber.signal_release();
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.pitch_gen.process_controls(control);
    }
}

impl<S, Pitch, Grabber> SIMDVoiceGenerator<S, SIMDSampleMono<S>>
    for SIMDMonoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleMono<S> {
        simd_invoke!(S, {
            let speed = self.pitch_gen.next_sample().0;
            let mut indexes = S::Vi32::zeroes();
            let mut fractionals = S::Vf32::zeroes();

            unsafe {
                for i in 0..S::Vf32::WIDTH {
                    let time = self.increment_time(speed.get_unchecked(i) as f64);
                    *indexes.get_unchecked_mut(i) = time as i32;
                    *fractionals.get_unchecked_mut(i) = (time % 1.0) as f32;
                }
            }

            let sample = self.grabber.get(indexes, fractionals);

            SIMDSampleMono(sample)
        })
    }
}

pub struct SIMDStereoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    grabber_left: Grabber,
    grabber_right: Grabber,

    pitch_gen: Pitch,

    time: f64,

    _s: PhantomData<S>,
}

impl<S, Pitch, Grabber> SIMDStereoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    pub fn new(grabber_left: Grabber, grabber_right: Grabber, pitch_gen: Pitch) -> Self {
        SIMDStereoVoiceSampler {
            grabber_left,
            grabber_right,
            pitch_gen,
            time: 0.0,
            _s: PhantomData,
        }
    }

    fn increment_time(&mut self, by: f64) -> f64 {
        let time = self.time;
        self.time += by;
        time
    }
}

impl<S, Pitch, Grabber> VoiceGeneratorBase for SIMDStereoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    #[inline(always)]
    fn ended(&self) -> bool {
        self.grabber_left.is_past_end(self.time) || self.grabber_right.is_past_end(self.time)
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.pitch_gen.signal_release(rel_type);
        self.grabber_left.signal_release();
        self.grabber_right.signal_release();
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.pitch_gen.process_controls(control);
    }
}

impl<S, Pitch, Grabber> SIMDVoiceGenerator<S, SIMDSampleStereo<S>>
    for SIMDStereoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleStereo<S> {
        simd_invoke!(S, {
            let speed = self.pitch_gen.next_sample().0;
            let mut indexes = S::Vi32::zeroes();
            let mut fractionals = S::Vf32::zeroes();

            unsafe {
                for i in 0..S::Vf32::WIDTH {
                    let time = self.increment_time(speed.get_unchecked(i) as f64);
                    *indexes.get_unchecked_mut(i) = time as i32;
                    *fractionals.get_unchecked_mut(i) = (time % 1.0) as f32;
                }
            }

            let left = self.grabber_left.get(indexes, fractionals);
            let right = self.grabber_right.get(indexes, fractionals);

            SIMDSampleStereo(left, right)
        })
    }
}
