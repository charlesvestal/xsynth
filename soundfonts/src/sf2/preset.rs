use super::{
    default_raw_envelope, instrument::Sf2Instrument, modulator::Sf2NoteModDestination,
    sample::Sf2Sample, zone::Sf2Zone, Sf2Preset, Sf2RawEnvelope, Sf2Region, Sf2SampleLinkType,
};
use crate::{convert_sample_index, sfz::AmpegEnvelopeParams, LoopMode};
use soundfont::Preset;
use std::{ops::RangeInclusive, sync::Arc};

#[derive(Clone, Debug)]
pub struct Sf2ParsedPreset {
    pub bank: u16,
    pub preset: u16,
    pub zones: Vec<Sf2Zone>,
}

impl Sf2ParsedPreset {
    pub fn parse_presets(presets: Vec<Preset>) -> Vec<Sf2ParsedPreset> {
        let mut presets_parsed: Vec<Sf2ParsedPreset> = Vec::new();

        for preset in presets {
            let zones = Sf2Zone::parse(preset.zones, false);

            presets_parsed.push(Sf2ParsedPreset {
                preset: preset.header.preset,
                bank: preset.header.bank,
                zones,
            });
        }

        presets_parsed
    }

    pub fn merge_presets(
        sample_data: Vec<Sf2Sample>,
        instruments: Vec<Sf2Instrument>,
        presets: Vec<Sf2ParsedPreset>,
        sample_rate: u32,
    ) -> Vec<Sf2Preset> {
        let mut out: Vec<Sf2Preset> = Vec::new();

        for preset in presets {
            let mut new_preset = Sf2Preset {
                preset: preset.preset,
                bank: preset.bank,
                regions: Vec::new(),
            };

            let mut regions = Vec::new();

            for zone in preset.zones {
                if let Some(instrument_idx) = zone.index {
                    let instrument = &instruments[instrument_idx as usize];

                    for subzone in &instrument.regions {
                        if let Some(sample_idx) = subzone.index {
                            let sample = &sample_data[sample_idx as usize];
                            if sample.data.is_empty() {
                                continue;
                            }
                            let keyrange = apply_fixed_value(
                                combine_ranges(
                                    zone.keyrange.clone().unwrap_or(0..=127),
                                    subzone.keyrange.clone().unwrap_or(0..=127),
                                ),
                                subzone.fixed_key.or(zone.fixed_key),
                            );
                            let velrange = apply_fixed_value(
                                combine_ranges(
                                    zone.velrange.clone().unwrap_or(0..=127),
                                    subzone.velrange.clone().unwrap_or(0..=127),
                                ),
                                subzone.fixed_velocity.or(zone.fixed_velocity),
                            );
                            let pan = sum_i16(zone.pan, subzone.pan).clamp(-500, 500);
                            let attenuation = sum_i16(zone.attenuation, subzone.attenuation);
                            let mut note_modulators = zone.note_modulators.clone();
                            note_modulators.extend(subzone.note_modulators.iter().copied());
                            let cutoff_cents = if zone.cutoff.is_some()
                                || subzone.cutoff.is_some()
                                || note_modulators.iter().any(|modulator| {
                                    modulator.destination()
                                        == Sf2NoteModDestination::InitialFilterFc
                                }) {
                                Some(merge_absolute_relative(13_500, zone.cutoff, subzone.cutoff))
                            } else {
                                None
                            };
                            let cutoff = cutoff_cents.map(|cents| raw_cutoff_to_hz(cents as f32));
                            let resonance =
                                sum_i16(zone.resonance, subzone.resonance) as f32 / 10.0;
                            let raw_envelope = merge_raw_envelope(&zone, subzone);
                            let offset = sum_sample_offset(
                                zone.offset,
                                zone.offset_coarse,
                                subzone.offset,
                                subzone.offset_coarse,
                            );
                            let end_offset = sum_sample_offset(
                                zone.end_offset,
                                zone.end_offset_coarse,
                                subzone.end_offset,
                                subzone.end_offset_coarse,
                            );
                            let loop_start_offset = sum_sample_offset(
                                zone.loop_start_offset,
                                zone.loop_start_offset_coarse,
                                subzone.loop_start_offset,
                                subzone.loop_start_offset_coarse,
                            );
                            let loop_end_offset = sum_sample_offset(
                                zone.loop_end_offset,
                                zone.loop_end_offset_coarse,
                                subzone.loop_end_offset,
                                subzone.loop_end_offset_coarse,
                            );
                            let sample_end_raw = (sample.original_length as i32 + end_offset)
                                .clamp(0, sample.original_length as i32)
                                as u32;
                            let region_sample = build_region_samples(sample, &sample_data);
                            let rendered_sample_end = region_sample
                                .iter()
                                .map(|sample| sample.len() as u32)
                                .min()
                                .unwrap_or(0);
                            let offset = convert_sample_index(
                                offset.clamp(0, sample_end_raw as i32) as u32,
                                sample.sample_rate,
                                sample_rate,
                            )
                            .min(rendered_sample_end);
                            let sample_end = convert_sample_index(
                                sample_end_raw,
                                sample.sample_rate,
                                sample_rate,
                            )
                            .min(rendered_sample_end);
                            let loop_start = convert_sample_index(
                                (sample.loop_start as i32 + loop_start_offset)
                                    .clamp(0, sample_end_raw as i32)
                                    as u32,
                                sample.sample_rate,
                                sample_rate,
                            )
                            .min(sample_end);
                            let loop_end = convert_sample_index(
                                (sample.loop_end as i32 + loop_end_offset)
                                    .clamp(0, sample_end_raw as i32)
                                    as u32,
                                sample.sample_rate,
                                sample_rate,
                            )
                            .min(sample_end);

                            let new_region = Sf2Region {
                                sample: region_sample,
                                sample_rate: sample.sample_rate,
                                velrange,
                                keyrange,
                                root_key: subzone
                                    .root_override
                                    .or(zone.root_override)
                                    .unwrap_or(sample.origpitch as i16)
                                    as u8,
                                volume: { 10f32.powf(-(attenuation as f32) / 200.0) },
                                pan,
                                loop_mode: zone
                                    .loop_mode
                                    .unwrap_or(subzone.loop_mode.unwrap_or(LoopMode::NoLoop)),
                                loop_start,
                                loop_end,
                                offset,
                                sample_end,
                                cutoff,
                                resonance,
                                fine_tune: sum_i16(zone.fine_tune, subzone.fine_tune)
                                    + sample.pitchadj as i16,
                                coarse_tune: sum_i16(zone.coarse_tune, subzone.coarse_tune),
                                ampeg_envelope: raw_envelope_to_ampeg(raw_envelope),
                                scale_tuning: subzone
                                    .scale_tuning
                                    .or(zone.scale_tuning)
                                    .unwrap_or(100),
                                exclusive_class: subzone.exclusive_class.or(zone.exclusive_class),
                                keynum_to_vol_env_hold: sum_i16(
                                    zone.keynum_to_vol_env_hold,
                                    subzone.keynum_to_vol_env_hold,
                                ),
                                keynum_to_vol_env_decay: sum_i16(
                                    zone.keynum_to_vol_env_decay,
                                    subzone.keynum_to_vol_env_decay,
                                ),
                                cutoff_cents,
                                raw_envelope,
                                note_modulators: Arc::from(note_modulators),
                            };

                            regions.push(new_region);
                        }
                    }
                }
            }

            for region in regions {
                new_preset.regions.push(region);
            }
            out.push(new_preset);
        }

        out
    }
}

fn combine_ranges<T: Ord + Copy>(
    r1: RangeInclusive<T>,
    r2: RangeInclusive<T>,
) -> RangeInclusive<T> {
    let start1 = r1.start();
    let start2 = r2.start();
    let end1 = r1.end();
    let end2 = r2.end();

    (*start1.max(start2))..=(*end1.min(end2))
}

fn apply_fixed_value<T: Ord + Copy>(
    range: RangeInclusive<T>,
    fixed: Option<T>,
) -> RangeInclusive<T> {
    if let Some(fixed) = fixed {
        combine_ranges(range, fixed..=fixed)
    } else {
        range
    }
}

fn sum_i16(a: Option<i16>, b: Option<i16>) -> i16 {
    a.unwrap_or(0) + b.unwrap_or(0)
}

fn sum_sample_offset(
    fine_a: Option<i16>,
    coarse_a: Option<i16>,
    fine_b: Option<i16>,
    coarse_b: Option<i16>,
) -> i32 {
    i32::from(sum_i16(fine_a, fine_b)) + i32::from(sum_i16(coarse_a, coarse_b)) * 32768
}

fn merge_absolute_relative(default: i16, preset: Option<i16>, instrument: Option<i16>) -> i32 {
    i32::from(instrument.unwrap_or(default)) + i32::from(preset.unwrap_or(0))
}

fn merge_raw_envelope(preset: &Sf2Zone, instrument: &Sf2Zone) -> Sf2RawEnvelope {
    let defaults = default_raw_envelope();
    Sf2RawEnvelope {
        delay_tc: merge_absolute_relative(
            defaults.delay_tc as i16,
            preset.env_delay,
            instrument.env_delay,
        ),
        attack_tc: merge_absolute_relative(
            defaults.attack_tc as i16,
            preset.env_attack,
            instrument.env_attack,
        ),
        hold_tc: merge_absolute_relative(
            defaults.hold_tc as i16,
            preset.env_hold,
            instrument.env_hold,
        ),
        decay_tc: merge_absolute_relative(
            defaults.decay_tc as i16,
            preset.env_decay,
            instrument.env_decay,
        ),
        sustain_cb: merge_absolute_relative(
            defaults.sustain_cb as i16,
            preset.env_sustain,
            instrument.env_sustain,
        ),
        release_tc: merge_absolute_relative(
            defaults.release_tc as i16,
            preset.env_release,
            instrument.env_release,
        ),
    }
}

fn raw_envelope_to_ampeg(raw: Sf2RawEnvelope) -> AmpegEnvelopeParams {
    AmpegEnvelopeParams {
        ampeg_start: 0.0,
        ampeg_delay: timecents_to_seconds(raw.delay_tc as f32),
        ampeg_attack: timecents_to_seconds(raw.attack_tc as f32),
        ampeg_hold: timecents_to_seconds(raw.hold_tc as f32),
        ampeg_decay: timecents_to_seconds(raw.decay_tc as f32),
        ampeg_sustain: sustain_cb_to_percent(raw.sustain_cb as f32),
        ampeg_release: timecents_to_seconds(raw.release_tc as f32),
        ampeg_vel2release: 0.0,
    }
}

fn raw_cutoff_to_hz(cents: f32) -> f32 {
    2f32.powf(cents.clamp(1500.0, 13_500.0) / 1200.0) * 8.176
}

fn timecents_to_seconds(timecents: f32) -> f32 {
    if timecents <= -32_768.0 {
        0.0
    } else {
        2f32.powf(timecents.clamp(-12_000.0, 8_000.0) / 1200.0)
    }
}

fn sustain_cb_to_percent(cb: f32) -> f32 {
    10f32.powf(-cb.max(0.0) / 200.0) * 100.0
}

fn build_region_samples(sample: &Sf2Sample, sample_data: &[Sf2Sample]) -> Arc<[Arc<[i16]>]> {
    match (sample.link_type, sample.linked_sample) {
        (Sf2SampleLinkType::Left, Some(linked)) => {
            if let Some(right) = sample_data.get(linked as usize) {
                if !right.data.is_empty() {
                    return Arc::new([sample.data.clone(), right.data.clone()]);
                }
            }
            Arc::new([sample.data.clone()])
        }
        (Sf2SampleLinkType::Right, Some(linked)) => {
            if let Some(left) = sample_data.get(linked as usize) {
                if !left.data.is_empty() {
                    return Arc::new([left.data.clone(), sample.data.clone()]);
                }
            }
            Arc::new([sample.data.clone()])
        }
        _ => Arc::new([sample.data.clone()]),
    }
}
