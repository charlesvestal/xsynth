#![allow(dead_code)]
#![allow(non_camel_case_types)] // For the SIMD library

mod envelopes;
pub(crate) use envelopes::*;

mod simd;
pub(crate) use simd::*;

mod simdvoice;
pub(crate) use simdvoice::*;

mod base;
pub(crate) use base::*;

mod squarewave;
#[allow(unused_imports)]
pub(crate) use squarewave::*;

mod channels;
#[allow(unused_imports)]
pub(crate) use channels::*;

mod constant;
pub(crate) use constant::*;

mod sampler;
pub(crate) use sampler::*;

mod control;
pub(crate) use control::*;

mod cutoff;
pub(crate) use cutoff::*;

mod oncc_amp;
pub(crate) use oncc_amp::*;

mod pan_oncc;
pub(crate) use pan_oncc::*;

mod lfo;
pub(crate) use lfo::*;

/// MOVE FORK: per-channel raw MIDI CC array. Updated by the channel
/// every `ControlEvent::Raw(cc, val)`. Voices clone the `Arc` at spawn
/// time and live `_oncc` generators (e.g. SIMDVoiceOnccAmp in a later
/// phase) read it under `Ordering::Relaxed`. ResetControl wipes back
/// to zeros. The array is fixed-size 128 — one byte per MIDI CC
/// number — so an Arc clone is just a refcount increment.
pub type CcState = std::sync::Arc<[std::sync::atomic::AtomicU8; 128]>;

/// MOVE FORK: build a fresh CcState with every CC at 0.
pub fn new_cc_state() -> CcState {
    std::sync::Arc::new(std::array::from_fn(|_| std::sync::atomic::AtomicU8::new(0)))
}

/// Options to modify the envelope of a voice.
#[derive(Copy, Clone)]
pub struct EnvelopeControlData {
    /// Controls the attack. Can take values from 0 to 128
    /// according to the MIDI CC spec.
    pub attack: Option<u8>,

    /// Controls the release. Can take values from 0 to 128
    /// according to the MIDI CC spec.
    pub release: Option<u8>,
}

/// How a voice should be released.
#[derive(Copy, Clone, PartialEq)]
pub enum ReleaseType {
    /// Standard release. Uses the voice's envelope.
    Standard,

    /// Kills the voice with a fadeout of 1ms.
    Kill,
}

/// Options to control the parameters of a voice.
#[derive(Copy, Clone)]
pub struct VoiceControlData {
    /// Pitch multiplier
    pub voice_pitch_multiplier: f32,

    /// Envelope control
    pub envelope: EnvelopeControlData,
}

impl VoiceControlData {
    pub fn new_defaults() -> Self {
        VoiceControlData {
            voice_pitch_multiplier: 1.0,
            envelope: EnvelopeControlData {
                attack: None,
                release: None,
            },
        }
    }
}

pub trait VoiceGeneratorBase: Sync + Send {
    fn ended(&self) -> bool;
    fn signal_release(&mut self, rel_type: ReleaseType);
    fn process_controls(&mut self, control: &VoiceControlData);
}

pub trait VoiceSampleGenerator: VoiceGeneratorBase {
    fn render_to(&mut self, buffer: &mut [f32]);
}

pub trait Voice: VoiceSampleGenerator + Send + Sync {
    fn is_releasing(&self) -> bool;
    fn is_killed(&self) -> bool;

    fn velocity(&self) -> u8;
    fn exclusive_class(&self) -> Option<u8>;

    /// MOVE FORK: set `is_releasing=true` on this voice WITHOUT
    /// triggering envelope release. Used for `trigger=release` voices
    /// at spawn time: they shouldn't be candidates for future NoteOff
    /// release (release_next_voice skips releasing voices), but their
    /// internal envelope must play its attack/sustain/release phases
    /// normally so the release-trigger sample plays its full authored
    /// duration. Default no-op so non-Voice implementors don't have to
    /// care.
    fn mark_as_release_trigger(&mut self) {}
}
