use std::{iter, ops::Deref, sync::{Arc, Mutex}};

use crate::{
    helpers::are_arc_vecs_equal,
    soundfont::SoundfontBase,
    voice::{CcState, Voice, VoiceControlData},
};

use super::voice_spawner::VoiceSpawnerMatrix;

/// MOVE FORK: deferred-drop sink. set_soundfonts pushes the OLD vec here
/// rather than letting it drop inline on the audio thread (which can take
/// 100 ms+ for a heavy DS library).
pub type SoundfontDropSink = Arc<Mutex<Vec<Arc<dyn SoundfontBase>>>>;

#[derive(Default, PartialEq, Eq, Clone, Copy, Debug)]
pub struct ProgramDescriptor {
    pub bank: u8,
    pub preset: u8,
}

pub struct ChannelSoundfont {
    soundfonts: Vec<Arc<dyn SoundfontBase>>,
    matrix: VoiceSpawnerMatrix,
    curr_program: ProgramDescriptor,
    drop_sink: Option<SoundfontDropSink>,
}

impl Deref for ChannelSoundfont {
    type Target = VoiceSpawnerMatrix;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.matrix
    }
}

impl ChannelSoundfont {
    pub fn new() -> Self {
        ChannelSoundfont {
            soundfonts: Vec::new(),
            matrix: VoiceSpawnerMatrix::new(),
            curr_program: Default::default(),
            drop_sink: None,
        }
    }

    pub fn set_drop_sink(&mut self, sink: SoundfontDropSink) {
        self.drop_sink = Some(sink);
    }

    pub fn set_soundfonts(&mut self, soundfonts: Vec<Arc<dyn SoundfontBase>>) {
        if !are_arc_vecs_equal(&self.soundfonts, &soundfonts) {
            let t_start = std::time::Instant::now();
            let old_count = self.soundfonts.len();
            let new_count = soundfonts.len();
            let old = std::mem::replace(&mut self.soundfonts, soundfonts);
            let t_after_replace = t_start.elapsed().as_micros();
            if let Some(sink) = &self.drop_sink {
                if let Ok(mut q) = sink.lock() {
                    q.extend(old);
                }
            }
            let t_after_sink = t_start.elapsed().as_micros();
            self.rebuild_matrix();
            let t_total = t_start.elapsed().as_micros();
            // MOVE FORK: write to a known file so the Move host can grep
            // it. stderr isn't captured anywhere we can read.
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/data/UserData/schwung/tmp/xsynth_debug.log")
            {
                let _ = writeln!(
                    f,
                    "[xsynth] set_soundfonts: old={} new={} replace={}us sink={}us rebuild={}us total={}us",
                    old_count, new_count, t_after_replace,
                    t_after_sink - t_after_replace,
                    t_total - t_after_sink, t_total
                );
            }
        }
    }

    pub fn change_program(&mut self, program: ProgramDescriptor) {
        if self.curr_program != program {
            self.curr_program = program;
            self.rebuild_matrix();
        }
    }

    fn rebuild_matrix(&mut self) {
        // MOVE FORK: instrument rebuild to find where it hangs/crashes.
        // Writes a heartbeat to xsynth_debug.log every ~quarter through.
        let t_start = std::time::Instant::now();
        let dbg = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/data/UserData/schwung/tmp/xsynth_debug.log")
            .ok();
        let log = |msg: String| {
            if let Some(mut f) = dbg.as_ref().and_then(|f| f.try_clone().ok()) {
                use std::io::Write;
                let _ = writeln!(f, "{}", msg);
            }
        };
        log(format!("[xsynth] rebuild_matrix: ENTER sf_count={}", self.soundfonts.len()));

        let bank = self.curr_program.bank;
        let preset = self.curr_program.preset;

        for k in 0..128u8 {
            for v in 0..128u8 {
                // Heartbeat every 4k iterations (4 times during full rebuild).
                let iter = (k as usize) * 128 + (v as usize);
                if iter % 4096 == 0 {
                    log(format!(
                        "[xsynth] rebuild_matrix: iter={} elapsed={}us",
                        iter,
                        t_start.elapsed().as_micros()
                    ));
                }
                let find_replacement_attack = || {
                    if bank == 128 {
                        self.soundfonts
                            .iter()
                            .map(|sf| sf.get_attack_voice_spawners_at(bank, 0, k, v))
                            .find(|vec| !vec.is_empty())
                    } else {
                        self.soundfonts
                            .iter()
                            .map(|sf| sf.get_attack_voice_spawners_at(0, preset, k, v))
                            .find(|vec| !vec.is_empty())
                    }
                };

                let attack_spawners = self
                    .soundfonts
                    .iter()
                    .map(|sf| sf.get_attack_voice_spawners_at(bank, preset, k, v))
                    .chain(iter::once_with(find_replacement_attack).flatten())
                    .find(|vec| !vec.is_empty())
                    .unwrap_or_default();

                let find_replacement_release = || {
                    if bank == 128 {
                        self.soundfonts
                            .iter()
                            .map(|sf| sf.get_release_voice_spawners_at(bank, 0, k, v))
                            .find(|vec| !vec.is_empty())
                    } else {
                        self.soundfonts
                            .iter()
                            .map(|sf| sf.get_release_voice_spawners_at(0, preset, k, v))
                            .find(|vec| !vec.is_empty())
                    }
                };

                let release_spawners = self
                    .soundfonts
                    .iter()
                    .map(|sf| sf.get_release_voice_spawners_at(bank, preset, k, v))
                    .chain(iter::once_with(find_replacement_release).flatten())
                    .find(|vec| !vec.is_empty())
                    .unwrap_or_default();

                self.matrix.set_spawners_attack(k, v, attack_spawners);
                self.matrix.set_spawners_release(k, v, release_spawners);
            }
        }
        log(format!("[xsynth] rebuild_matrix: EXIT elapsed={}us", t_start.elapsed().as_micros()));
    }

    pub fn spawn_voices_attack<'a>(
        &'a self,
        control: &'a VoiceControlData,
        cc_state: &'a CcState,
        key: u8,
        vel: u8,
    ) -> impl Iterator<Item = Box<dyn Voice>> + 'a {
        self.matrix.spawn_voices_attack(control, cc_state, key, vel)
    }

    pub fn spawn_voices_release<'a>(
        &'a self,
        control: &'a VoiceControlData,
        cc_state: &'a CcState,
        key: u8,
        vel: u8,
    ) -> impl Iterator<Item = Box<dyn Voice>> + 'a {
        self.matrix.spawn_voices_release(control, cc_state, key, vel)
    }

    pub fn exclusive_classes_attack<'a>(
        &'a self,
        key: u8,
        vel: u8,
    ) -> impl Iterator<Item = u8> + 'a {
        self.matrix.exclusive_classes_attack(key, vel)
    }
}
