use std::sync::Arc;

use crate::soundfont::SoundfontBase;

/// MIDI events for a single key in a channel.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum KeyNoteEvent {
    /// Starts a new note voice with a velocity
    On(u8),

    /// Signals off to a note voice
    Off,

    /// Signals off to all note voices
    AllOff,

    /// Kills all note voices without decay
    AllKilled,
}

/// Events to modify parameters of a channel.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum ChannelConfigEvent {
    /// Sets the soundfonts for the channel
    #[cfg_attr(feature = "serde", serde(skip))]
    SetSoundfonts(Vec<Arc<dyn SoundfontBase>>),

    /// Sets the layer count for the soundfont
    SetLayerCount(Option<usize>),

    /// MOVE FORK: polyphony (note-group) cap for the channel. When the
    /// number of active note groups exceeds this at the start of a render
    /// block, the oldest releasing group is dropped whole (all voices that
    /// share its group id, even across keys). Falls back to dropping the
    /// oldest non-releasing group as a last resort. `None` disables the
    /// cap. A "group" is one note-on's worth of voices (e.g. WörliTzer's 5
    /// layers per key = one group of 5 voices).
    SetPolyphonyCap(Option<usize>),

    /// MOVE FORK: max NoteOn events processed per render block. Spawning
    /// a voice is ~12 µs (alloc + filter/envelope init + first sample
    /// reads); a quantized 10-note chord landing in one block adds 50
    /// voices and ~600 µs of spawn work on top of render. With this set,
    /// surplus NoteOn events stay in the per-key event cache and fire on
    /// subsequent blocks (one block ≈ 2.9 ms at 44.1 kHz / 128 frames —
    /// well below human latency perception). `None` disables the limit.
    SetSpawnBurstLimit(Option<usize>),

    /// Controls whether the channel will be standard or percussion.
    /// Setting to `true` will make the channel only use percussion patches.
    SetPercussionMode(bool),

    /// MOVE FORK / Phase 9 prototype: install a fundsp reverb on the
    /// channel's post-mix bus. Pass `(room_size_seconds, time, damp)`
    /// in fundsp `reverb_stereo` units. `None` removes any existing
    /// reverb. Sample rate is configured on install.
    #[cfg_attr(feature = "serde", serde(skip))]
    SetReverb(Option<(f32, f32, f32)>),

    /// MOVE FORK / Phase 9 prototype: dry/wet mix for the channel
    /// reverb (0 = dry, 1 = wet). Reverb processing is skipped
    /// entirely when wet == 0.
    SetReverbWet(f32),

    /// MOVE FORK / Phase 9: install/remove the hand-rolled stereo
    /// feedback delay on the channel post-mix. `Some((time, fb))` =
    /// install with delay time in seconds and feedback amount 0..0.95.
    /// `None` removes (drops the ring buffer).
    SetDelay(Option<(f32, f32)>),

    /// MOVE FORK / Phase 9: live delay time update (seconds, clamped
    /// to 0..2). No-op when no delay is installed.
    SetDelayTime(f32),

    /// MOVE FORK / Phase 9: live feedback update (0..0.95).
    SetDelayFeedback(f32),

    /// MOVE FORK / Phase 9: dry/wet mix (0..1). Zero skips processing.
    SetDelayMix(f32),

    /// MOVE FORK / Phase 12: install/remove stereo chorus on the
    /// channel post-mix. `Some((rate, depth))` = install with the
    /// given LFO rate (Hz) and depth (0..1 fraction of center delay).
    /// `None` removes (drops the ring buffers).
    SetChorus(Option<(f32, f32)>),
    /// MOVE FORK / Phase 12: live rate update.
    SetChorusRate(f32),
    /// MOVE FORK / Phase 12: live depth update.
    SetChorusDepth(f32),
    /// MOVE FORK / Phase 12: dry/wet mix (crossfade, 0..1).
    SetChorusMix(f32),
}

/// MIDI events for a channel.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum ChannelAudioEvent {
    /// Starts a new note voice
    NoteOn { key: u8, vel: u8 },

    /// Signals off to a note voice
    NoteOff { key: u8 },

    /// Signal off to all voices
    AllNotesOff,

    /// Kill all voices without decay
    AllNotesKilled,

    /// Resets all CC to their default values
    ResetControl,

    /// Control event for the channel
    Control(ControlEvent),

    /// Program change event
    ProgramChange(u8),

    /// System reset
    SystemReset,
}

/// Wrapper enum for various events for a channel.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum ChannelEvent {
    /// Audio event
    Audio(ChannelAudioEvent),

    /// Configuration event for the channel
    Config(ChannelConfigEvent),
}

/// MIDI control events for a channel.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum ControlEvent {
    /// A raw control change event
    Raw(u8, u8),

    /// The pitch bend strength, in tones
    PitchBendSensitivity(f32),

    /// The pitch bend value, between -1 and 1
    PitchBendValue(f32),

    /// The pitch bend, product of value * sensitivity
    PitchBend(f32),

    /// Fine tune value in cents
    FineTune(f32),

    /// Coarse tune value in semitones
    CoarseTune(f32),
}
