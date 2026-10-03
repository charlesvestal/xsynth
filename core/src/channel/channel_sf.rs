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

/// MOVE FORK / 2026-10-03: where a channel puts the spawner matrix it
/// replaced, so the ~16k small Vecs are freed off the audio thread.
pub type MatrixDropSink = Arc<Mutex<Vec<Box<VoiceSpawnerMatrix>>>>;

/// MOVE FORK / 2026-10-03: a spawner matrix built AHEAD of time, off the
/// audio thread, for `program`. Carried by
/// `ChannelConfigEvent::SetSoundfontsPrebuilt`. Events are cloned when a
/// channel group broadcasts them, so the matrix sits in a shared slot and the
/// first channel to apply the event takes it; a clone finds the slot empty
/// and falls back to an in-place rebuild.
#[derive(Clone)]
pub struct PrebuiltMatrix {
    slot: Arc<Mutex<Option<Box<VoiceSpawnerMatrix>>>>,
    pub program: ProgramDescriptor,
    retired: MatrixDropSink,
}

impl PrebuiltMatrix {
    pub fn new(matrix: VoiceSpawnerMatrix, program: ProgramDescriptor, retired: MatrixDropSink) -> Self {
        PrebuiltMatrix { slot: Arc::new(Mutex::new(Some(Box::new(matrix)))), program, retired }
    }
}

impl std::fmt::Debug for PrebuiltMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PrebuiltMatrix({:?})", self.program)
    }
}

/// MOVE FORK / 2026-10-03: build the spawner matrix for `soundfonts` at
/// `program` anywhere -- in particular on a loader thread, so that applying a
/// new instrument on the audio thread is a pointer swap instead of a
/// 128 x 128 walk over every region (measured at ~41 ms on a Move for a
/// 57 MB SFZ, in one render block).
pub fn build_spawner_matrix(
    soundfonts: &[Arc<dyn SoundfontBase>],
    program: ProgramDescriptor,
) -> VoiceSpawnerMatrix {
    let mut matrix = VoiceSpawnerMatrix::new();
    fill_matrix(&mut matrix, soundfonts, program);
    matrix
}

#[derive(Default, PartialEq, Eq, Clone, Copy, Debug)]
pub struct ProgramDescriptor {
    pub bank: u8,
    pub preset: u8,
}

pub struct ChannelSoundfont {
    soundfonts: Vec<Arc<dyn SoundfontBase>>,
    /// Boxed so a prebuilt matrix swaps in with no copy and no free.
    matrix: Box<VoiceSpawnerMatrix>,
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
            matrix: Box::new(VoiceSpawnerMatrix::new()),
            curr_program: Default::default(),
            drop_sink: None,
        }
    }

    pub fn set_drop_sink(&mut self, sink: SoundfontDropSink) {
        self.drop_sink = Some(sink);
    }

    pub fn set_soundfonts(&mut self, soundfonts: Vec<Arc<dyn SoundfontBase>>) {
        if !are_arc_vecs_equal(&self.soundfonts, &soundfonts) {
            let old = std::mem::replace(&mut self.soundfonts, soundfonts);
            if let Some(sink) = &self.drop_sink {
                if let Ok(mut q) = sink.lock() {
                    q.extend(old);
                }
            }
            // (The MOVE FORK timing log that used to sit here opened a file
            // on every call -- on the audio thread. Removed 2026-10-03.)
            self.rebuild_matrix();
        }
    }

    /// MOVE FORK / 2026-10-03: install soundfonts together with a matrix
    /// built for them off the audio thread. A swap, not a rebuild: the old
    /// matrix goes to the retire sink, the old soundfonts to the drop sink.
    /// Falls back to `set_soundfonts` (an in-place rebuild) if the matrix
    /// was built for a different program than the channel now has, or if
    /// another channel already took it.
    pub fn set_soundfonts_prebuilt(
        &mut self,
        soundfonts: Vec<Arc<dyn SoundfontBase>>,
        prebuilt: PrebuiltMatrix,
    ) {
        let taken = match prebuilt.slot.try_lock() {
            Ok(mut g) => g.take(),
            Err(_) => None,
        };
        let Some(matrix) = taken else {
            self.set_soundfonts(soundfonts);
            return;
        };
        if prebuilt.program != self.curr_program {
            if let Ok(mut q) = prebuilt.retired.lock() {
                q.push(matrix);
            }
            self.set_soundfonts(soundfonts);
            return;
        }
        let old = std::mem::replace(&mut self.soundfonts, soundfonts);
        if let Some(sink) = &self.drop_sink {
            if let Ok(mut q) = sink.lock() {
                q.extend(old);
            }
        }
        let old_matrix = std::mem::replace(&mut self.matrix, matrix);
        if let Ok(mut q) = prebuilt.retired.lock() {
            q.push(old_matrix);
        }
    }

    pub fn change_program(&mut self, program: ProgramDescriptor) {
        if self.curr_program != program {
            self.curr_program = program;
            self.rebuild_matrix();
        }
    }

    fn rebuild_matrix(&mut self) {
        fill_matrix(&mut self.matrix, &self.soundfonts, self.curr_program);
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

/// The 128 x 128 key/velocity walk, shared by the in-place rebuild and
/// `build_spawner_matrix`.
fn fill_matrix(
    matrix: &mut VoiceSpawnerMatrix,
    soundfonts: &[Arc<dyn SoundfontBase>],
    program: ProgramDescriptor,
) {
    let bank = program.bank;
    let preset = program.preset;

        for k in 0..128u8 {
            for v in 0..128u8 {
                let find_replacement_attack = || {
                    if bank == 128 {
                        soundfonts
                            .iter()
                            .map(|sf| sf.get_attack_voice_spawners_at(bank, 0, k, v))
                            .find(|vec| !vec.is_empty())
                    } else {
                        soundfonts
                            .iter()
                            .map(|sf| sf.get_attack_voice_spawners_at(0, preset, k, v))
                            .find(|vec| !vec.is_empty())
                    }
                };

                let attack_spawners = soundfonts
                    .iter()
                    .map(|sf| sf.get_attack_voice_spawners_at(bank, preset, k, v))
                    .chain(iter::once_with(find_replacement_attack).flatten())
                    .find(|vec| !vec.is_empty())
                    .unwrap_or_default();

                let find_replacement_release = || {
                    if bank == 128 {
                        soundfonts
                            .iter()
                            .map(|sf| sf.get_release_voice_spawners_at(bank, 0, k, v))
                            .find(|vec| !vec.is_empty())
                    } else {
                        soundfonts
                            .iter()
                            .map(|sf| sf.get_release_voice_spawners_at(0, preset, k, v))
                            .find(|vec| !vec.is_empty())
                    }
                };

                let release_spawners = soundfonts
                    .iter()
                    .map(|sf| sf.get_release_voice_spawners_at(bank, preset, k, v))
                    .chain(iter::once_with(find_replacement_release).flatten())
                    .find(|vec| !vec.is_empty())
                    .unwrap_or_default();

                matrix.set_spawners_attack(k, v, attack_spawners);
                matrix.set_spawners_release(k, v, release_spawners);
            }
        }
}
