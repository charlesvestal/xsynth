use std::sync::{atomic::AtomicU64, Arc};

use crate::AudioStreamParams;

use super::{
    channel_sf::{ChannelSoundfont, ProgramDescriptor},
    ChannelConfigEvent,
};

/// Holds the statistics for an instance of VoiceChannel.
#[derive(Debug, Clone)]
pub struct VoiceChannelStats {
    pub(super) voice_counter: Arc<AtomicU64>,
}

/// Reads the statistics of an instance of VoiceChannel in a usable way.
pub struct VoiceChannelStatsReader {
    stats: VoiceChannelStats,
}

#[derive(Debug, Clone)]
pub struct VoiceChannelConst {
    pub stream_params: AudioStreamParams,
}

pub struct VoiceChannelParams {
    pub stats: VoiceChannelStats,
    pub layers: Option<usize>,
    /// MOVE FORK: polyphony (note-group) cap. See `ChannelConfigEvent::SetPolyphonyCap`.
    pub polyphony_cap: Option<usize>,
    /// MOVE FORK: max NoteOn events drained per render block. Defers
    /// surplus events to subsequent blocks. See `SetSpawnBurstLimit`.
    pub spawn_burst_limit: Option<usize>,
    pub channel_sf: ChannelSoundfont,
    pub program: ProgramDescriptor,
    pub constant: VoiceChannelConst,
}

impl VoiceChannelStats {
    pub fn new() -> Self {
        let voice_counter = Arc::new(AtomicU64::new(0));
        Self { voice_counter }
    }
}

impl Default for VoiceChannelStats {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceChannelParams {
    pub fn new(stream_params: AudioStreamParams) -> Self {
        let channel_sf = ChannelSoundfont::new();

        Self {
            stats: VoiceChannelStats::new(),
            layers: Some(4),
            polyphony_cap: None,
            spawn_burst_limit: None,
            channel_sf,
            program: Default::default(),
            constant: VoiceChannelConst { stream_params },
        }
    }

    pub fn process_config_event(&mut self, event: ChannelConfigEvent) {
        match event {
            ChannelConfigEvent::SetSoundfonts(soundfonts) => {
                self.channel_sf.set_soundfonts(soundfonts)
            }
            ChannelConfigEvent::SetLayerCount(count) => {
                self.layers = count;
            }
            ChannelConfigEvent::SetPolyphonyCap(cap) => {
                self.polyphony_cap = cap;
            }
            ChannelConfigEvent::SetSpawnBurstLimit(limit) => {
                self.spawn_burst_limit = limit;
            }
            ChannelConfigEvent::SetPercussionMode(set) => {
                if set {
                    self.program.bank = 128;
                } else {
                    self.program.bank = 0;
                }
                self.channel_sf.change_program(self.program);
            }
            // MOVE FORK / Phase 9: reverb events are intercepted by
            // VoiceChannel before reaching here. If one slips through
            // (e.g. via a different code path), treat as no-op.
            ChannelConfigEvent::SetReverb(_)
            | ChannelConfigEvent::SetReverbWet(_) => {}
        }
    }

    pub fn set_bank(&mut self, bank: u8) {
        if self.program.bank != 128 {
            self.program.bank = bank.min(127);
        }
    }

    pub fn set_preset(&mut self, preset: u8) {
        self.program.preset = preset.min(127);
    }

    pub fn load_program(&mut self) {
        self.channel_sf.change_program(self.program);
    }
}

impl VoiceChannelStatsReader {
    pub(super) fn new(stats: VoiceChannelStats) -> Self {
        Self { stats }
    }

    /// The active voice count of the VoiceChannel.
    pub fn voice_count(&self) -> u64 {
        self.stats
            .voice_counter
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}
