use std::{marker::PhantomData, ops::Mul, sync::Arc};

use simdeez::Simd;

use crate::{
    effects::BiQuadFilter,
    voice::{
        BufferSampler, SIMDSample, SIMDSampleGrabber, SIMDSampleMono, SIMDSampleStereo,
        SIMDStereoVoiceCutoffLive, SIMDVoiceGenerator,
    },
    AudioStreamParams,
};
use crate::{
    voice::VoiceControlData,
    voice::{
        BufferSamplers, CcState, EnvelopeParameters, SIMDConstant,
        SIMDLinearSampleGrabber, SIMDNearestSampleGrabber, SIMDStereoVoice, SIMDStereoVoiceSampler,
        SIMDVoiceControl, SIMDVoiceEnvelope, SIMDVoiceLfoAmp, SIMDVoiceLfoPitch, SIMDVoiceOnccAmp, SIMDVoicePan, SampleReader,
        SampleReaderLoop, SampleReaderLoopSustain, SampleReaderNoLoop, Voice, VoiceBase,
        VoiceCombineSIMD,
    },
};

use xsynth_soundfonts::LoopMode;

use crate::soundfont::{Interpolator, LoopParams, SampleStorage, SampleVoiceSpawnerParams, VoiceSpawner};

pub struct StereoSampledVoiceSpawner<S: 'static + Simd + Send + Sync> {
    speed_mult: f32,
    filter: Option<BiQuadFilter>,
    /// MOVE FORK: filter params kept around for live coefficient
    /// recomputation by SIMDStereoVoiceCutoffLive.
    filter_type: xsynth_soundfonts::FilterType,
    base_cutoff: f32,
    base_resonance_db: f32,
    loop_params: LoopParams,
    amp: f32,
    pan: f32,
    volume_envelope_params: Arc<EnvelopeParameters>,
    samples: Arc<[Arc<SampleStorage>]>,
    interpolator: Interpolator,
    exclusive_class: Option<u8>,
    vel: u8,
    stream_params: AudioStreamParams,
    /// MOVE FORK: live `volume_oncc` bindings — when non-empty the voice
    /// gets a SIMDVoiceOnccAmp stage that polls the channel CC array.
    volume_oncc: Arc<[(u8, f32)]>,
    /// MOVE FORK: live `pan_oncc` bindings — fed to SIMDVoicePan which
    /// replaces the static SIMDConstantStereo pan stage.
    pan_oncc: Arc<[(u8, f32)]>,
    /// MOVE FORK: live `cutoff_oncc` bindings — fed to
    /// SIMDStereoVoiceCutoffLive which recomputes biquad coefficients
    /// per `RECOMPUTE_INTERVAL`.
    cutoff_oncc: Arc<[(u8, f32)]>,
    /// MOVE FORK: live `resonance_oncc` bindings.
    resonance_oncc: Arc<[(u8, f32)]>,
    /// MOVE FORK / Phase 6: per-CC curve_id for each `_oncc` family.
    volume_curvecc: Arc<[(u8, u8)]>,
    pan_curvecc: Arc<[(u8, u8)]>,
    cutoff_curvecc: Arc<[(u8, u8)]>,
    resonance_curvecc: Arc<[(u8, u8)]>,
    /// MOVE FORK / Phase 6: shared curve table map.
    curves: Arc<std::collections::HashMap<u8, [f32; 128]>>,
    /// MOVE FORK / Phase 11: amp LFO params.
    amp_lfo_freq: f32,
    amp_lfo_depth: f32,
    amp_lfo_freq_oncc: Arc<[(u8, f32)]>,
    amp_lfo_depth_oncc: Arc<[(u8, f32)]>,
    fil_lfo_freq: f32,
    fil_lfo_depth: f32,
    fil_lfo_freq_oncc: Arc<[(u8, f32)]>,
    fil_lfo_depth_oncc: Arc<[(u8, f32)]>,
    pan_lfo_freq: f32,
    pan_lfo_depth: f32,
    pan_lfo_freq_oncc: Arc<[(u8, f32)]>,
    pan_lfo_depth_oncc: Arc<[(u8, f32)]>,
    fileg_attack: f32,
    fileg_decay: f32,
    fileg_sustain: f32,
    fileg_release: f32,
    fileg_depth: f32,
    pitch_lfo_freq: f32,
    pitch_lfo_depth: f32,
    pitch_lfo_freq_oncc: Arc<[(u8, f32)]>,
    pitch_lfo_depth_oncc: Arc<[(u8, f32)]>,
    _s: PhantomData<S>,
}

impl<S: Simd + Send + Sync> StereoSampledVoiceSpawner<S> {
    pub fn new(
        params: &SampleVoiceSpawnerParams,
        vel: u8,
        stream_params: AudioStreamParams,
    ) -> Self {
        let amp = params.volume;

        let filter = params.cutoff.map(|cutoff| {
            BiQuadFilter::new(
                params.filter_type,
                cutoff,
                stream_params.sample_rate as f32,
                Some(params.resonance),
            )
        });

        Self {
            speed_mult: params.speed_mult,
            filter,
            filter_type: params.filter_type,
            base_cutoff: params.cutoff.unwrap_or(22000.0),
            base_resonance_db: params.base_resonance_db,
            loop_params: params.loop_params.clone(),
            amp,
            pan: params.pan,
            volume_envelope_params: params.envelope.clone(),
            samples: params.sample.clone(),
            interpolator: params.interpolator,
            exclusive_class: params.exclusive_class,
            vel,
            stream_params,
            volume_oncc: params.volume_oncc.clone(),
            pan_oncc: params.pan_oncc.clone(),
            cutoff_oncc: params.cutoff_oncc.clone(),
            resonance_oncc: params.resonance_oncc.clone(),
            volume_curvecc: params.volume_curvecc.clone(),
            pan_curvecc: params.pan_curvecc.clone(),
            cutoff_curvecc: params.cutoff_curvecc.clone(),
            resonance_curvecc: params.resonance_curvecc.clone(),
            curves: params.curves.clone(),
            amp_lfo_freq: params.amp_lfo_freq,
            amp_lfo_depth: params.amp_lfo_depth,
            amp_lfo_freq_oncc: params.amp_lfo_freq_oncc.clone(),
            amp_lfo_depth_oncc: params.amp_lfo_depth_oncc.clone(),
            fil_lfo_freq: params.fil_lfo_freq,
            fil_lfo_depth: params.fil_lfo_depth,
            fil_lfo_freq_oncc: params.fil_lfo_freq_oncc.clone(),
            fil_lfo_depth_oncc: params.fil_lfo_depth_oncc.clone(),
            pan_lfo_freq: params.pan_lfo_freq,
            pan_lfo_depth: params.pan_lfo_depth,
            pan_lfo_freq_oncc: params.pan_lfo_freq_oncc.clone(),
            pan_lfo_depth_oncc: params.pan_lfo_depth_oncc.clone(),
            fileg_attack:  params.fileg_attack,
            fileg_decay:   params.fileg_decay,
            fileg_sustain: params.fileg_sustain,
            fileg_release: params.fileg_release,
            fileg_depth:   params.fileg_depth,
            pitch_lfo_freq:  params.pitch_lfo_freq,
            pitch_lfo_depth: params.pitch_lfo_depth,
            pitch_lfo_freq_oncc:  params.pitch_lfo_freq_oncc.clone(),
            pitch_lfo_depth_oncc: params.pitch_lfo_depth_oncc.clone(),
            _s: PhantomData,
        }
    }

    fn begin_voice(&self, control: &VoiceControlData, cc_state: &CcState) -> Box<dyn Voice> {
        // Currently there's only the f32 buffer samples, more could be added in the future.
        #[allow(clippy::redundant_closure)]
        self.make_sample_reader(control, cc_state, |s| BufferSamplers::new_f32(s))
    }

    fn make_sample_reader<BS: 'static + BufferSampler>(
        &self,
        control: &VoiceControlData,
        cc_state: &CcState,
        make_bs: impl Fn(Arc<SampleStorage>) -> BS,
    ) -> Box<dyn Voice> {
        match self.loop_params.mode {
            LoopMode::LoopContinuous => self.make_sample_grabber(control, cc_state, move |s| {
                SampleReaderLoop::new(make_bs(s), self.loop_params.clone())
            }),
            LoopMode::LoopSustain => self.make_sample_grabber(control, cc_state, move |s| {
                SampleReaderLoopSustain::new(make_bs(s), self.loop_params.clone())
            }),
            LoopMode::NoLoop | LoopMode::OneShot => self.make_sample_grabber(control, cc_state, move |s| {
                SampleReaderNoLoop::new(make_bs(s), self.loop_params.clone())
            }),
        }
    }

    fn make_sample_grabber<SR: 'static + SampleReader>(
        &self,
        control: &VoiceControlData,
        cc_state: &CcState,
        make_bs: impl Fn(Arc<SampleStorage>) -> SR,
    ) -> Box<dyn Voice> {
        match self.interpolator {
            Interpolator::Nearest => {
                self.generate_sampler(control, cc_state, |s| SIMDNearestSampleGrabber::new(make_bs(s)))
            }
            Interpolator::Linear => {
                self.generate_sampler(control, cc_state, |s| SIMDLinearSampleGrabber::new(make_bs(s)))
            }
        }
    }

    fn generate_sampler<SG: 'static + SIMDSampleGrabber<S>>(
        &self,
        control: &VoiceControlData,
        cc_state: &CcState,
        make_sampler: impl Fn(Arc<SampleStorage>) -> SG,
    ) -> Box<dyn Voice> {
        let left = make_sampler(self.samples[0].clone());
        let right = make_sampler(self.samples[1].clone());

        let pitch_fac = self.create_pitch_fac(control, cc_state);

        let sampler = SIMDStereoVoiceSampler::new(left, right, pitch_fac);
        self.apply_voice_params(sampler, control, cc_state)
    }

    fn apply_velocity<Gen, Sample>(&self, gen: Gen) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleMono<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let amp = SIMDConstant::<S>::new(self.amp);
        let amp = VoiceCombineSIMD::mult(amp, gen);
        amp
    }

    /// MOVE FORK: pan stage is always a SIMDVoicePan — when bindings are
    /// empty the generator's `static_pan` flag caches the gains once and
    /// the cost matches the prior SIMDConstantStereo. With bindings, the
    /// stage polls the CC atomic array on a counter and recomputes the
    /// cos/sin equal-power gains live.
    fn apply_pan<Gen, Sample>(&self, gen: Gen, cc_state: &CcState) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleStereo<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let pan_gen = SIMDVoicePan::<S>::new(
            cc_state.clone(),
            self.pan_oncc.clone(),
            self.pan_curvecc.clone(),
            self.curves.clone(),
            self.pan,
            self.stream_params.sample_rate as f32,
            self.pan_lfo_freq,
            self.pan_lfo_depth,
            self.pan_lfo_freq_oncc.clone(),
            self.pan_lfo_depth_oncc.clone(),
        );
        VoiceCombineSIMD::mult(pan_gen, gen)
    }

    fn create_pitch_fac(
        &self,
        control: &VoiceControlData,
        cc_state: &CcState,
    ) -> impl SIMDVoiceGenerator<S, SIMDSampleMono<S>> {
        let pitch_fac = SIMDConstant::<S>::new(self.speed_mult);
        let pitch_multiplier = SIMDVoiceControl::new(control, |vc| vc.voice_pitch_multiplier);
        let pitch_fac = VoiceCombineSIMD::mult(pitch_fac, pitch_multiplier);
        // MOVE FORK / Phase 11: pitch LFO (vibrato). Multiplies a
        // sin-modulated cents factor into the read-rate.
        let pitch_lfo = SIMDVoiceLfoPitch::<S>::new(
            cc_state.clone(),
            self.pitch_lfo_freq,
            self.pitch_lfo_depth,
            self.stream_params.sample_rate as f32,
            self.pitch_lfo_freq_oncc.clone(),
            self.pitch_lfo_depth_oncc.clone(),
        );
        VoiceCombineSIMD::mult(pitch_fac, pitch_lfo)
    }

    fn apply_envelope<Gen, Sample>(
        &self,
        gen: Gen,
        control: &VoiceControlData,
    ) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleMono<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let modified_params = SIMDVoiceEnvelope::<S>::get_modified_envelope(
            *self.volume_envelope_params.clone(),
            control.envelope,
            self.stream_params.sample_rate as f32,
        );

        let allow_release = self.loop_params.mode != LoopMode::OneShot;

        let volume_envelope = SIMDVoiceEnvelope::new(
            *self.volume_envelope_params.clone(),
            modified_params,
            allow_release,
            self.stream_params.sample_rate as f32,
        );

        let amp = VoiceCombineSIMD::mult(volume_envelope, gen);
        amp
    }

    fn convert_to_voice<Gen>(&self, gen: Gen) -> Box<dyn Voice>
    where
        Gen: 'static + SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
    {
        let flattened = SIMDStereoVoice::new(gen);
        let base = VoiceBase::new(self.vel, self.exclusive_class(), flattened);

        Box::new(base)
    }

    fn apply_voice_params<Gen>(
        &self,
        gen: Gen,
        control: &VoiceControlData,
        cc_state: &CcState,
    ) -> Box<dyn Voice>
    where
        Gen: 'static + SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
    {
        let gen = self.apply_velocity(gen);
        let gen = self.apply_volume_oncc(gen, cc_state);
        let gen = self.apply_amp_lfo(gen, cc_state);
        let gen = self.apply_pan(gen, cc_state);
        let gen = self.apply_envelope(gen, control);

        self.apply_cutoff_effect(gen, cc_state)
    }

    /// MOVE FORK: live volume modulation via SFZ `volume_oncc<N>=<dB>`.
    /// The SIMDVoiceOnccAmp stage is always inserted to keep the
    /// generator chain monomorphic; when bindings are empty its
    /// `static_amp` short-circuit returns a permanent `set1(1.0)` and
    /// the cost matches the velocity `SIMDConstant` stage (one SIMD
    /// multiply per sample-width). On regions with non-empty bindings
    /// the stage polls the channel's CC atomic array on a counter
    /// (RECOMPUTE_INTERVAL — see voice/oncc_amp.rs) and produces the
    /// per-voice amp multiplier `db_to_amp(Σ delta·cc/127)`.
    /// MOVE FORK / Phase 11: amp LFO (sine tremolo) stage. Always
    /// inserted; the SIMDVoiceLfoAmp short-circuits to constant 1.0
    /// when freq or depth is zero (same cost as a static SIMDConstant
    /// stage in that case).
    fn apply_amp_lfo<Gen, Sample>(&self, gen: Gen, cc_state: &CcState) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleMono<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let lfo = SIMDVoiceLfoAmp::<S>::new(
            cc_state.clone(),
            self.amp_lfo_freq, self.amp_lfo_depth,
            self.stream_params.sample_rate as f32,
            self.amp_lfo_freq_oncc.clone(),
            self.amp_lfo_depth_oncc.clone(),
        );
        VoiceCombineSIMD::mult(lfo, gen)
    }

    fn apply_volume_oncc<Gen, Sample>(
        &self,
        gen: Gen,
        cc_state: &CcState,
    ) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleMono<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let oncc = SIMDVoiceOnccAmp::<S>::new(
            cc_state.clone(),
            self.volume_oncc.clone(),
            self.volume_curvecc.clone(),
            self.curves.clone(),
        );
        VoiceCombineSIMD::mult(oncc, gen)
    }

    /// MOVE FORK: cutoff stage uses SIMDStereoVoiceCutoffLive when a
    /// filter is present. The Live variant always wraps the filter and
    /// short-circuits coefficient recomputation when both `cutoff_oncc`
    /// and `resonance_oncc` are empty — matching the prior static
    /// SIMDStereoVoiceCutoff cost.
    fn apply_cutoff_effect(
        &self,
        gen: impl 'static + SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
        cc_state: &CcState,
    ) -> Box<dyn Voice> {
        if let Some(filter) = &self.filter {
            let gen = SIMDStereoVoiceCutoffLive::new(
                gen,
                filter,
                cc_state.clone(),
                self.cutoff_oncc.clone(),
                self.resonance_oncc.clone(),
                self.cutoff_curvecc.clone(),
                self.resonance_curvecc.clone(),
                self.curves.clone(),
                self.fil_lfo_freq,
                self.fil_lfo_depth,
                self.fil_lfo_freq_oncc.clone(),
                self.fil_lfo_depth_oncc.clone(),
                self.fileg_attack,
                self.fileg_decay,
                self.fileg_sustain,
                self.fileg_release,
                self.fileg_depth,
                self.filter_type,
                self.stream_params.sample_rate as f32,
                self.base_cutoff,
                self.base_resonance_db,
            );
            self.convert_to_voice(gen)
        } else {
            self.convert_to_voice(gen)
        }
    }
}

impl<S: 'static + Sync + Send + Simd> VoiceSpawner for StereoSampledVoiceSpawner<S> {
    fn spawn_voice(&self, control: &VoiceControlData, cc_state: &CcState) -> Box<dyn Voice> {
        self.begin_voice(control, cc_state)
    }

    fn exclusive_class(&self) -> Option<u8> {
        self.exclusive_class
    }
}
