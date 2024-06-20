import os

import librosa.display
import pretty_midi
from matplotlib import pyplot as plt
from pretty_midi import PrettyMIDI


def save_midi(path, pitches) -> PrettyMIDI:
    file = PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    pitch_dict = {}

    for timestep in range(len(pitches)):
        for pitch in range(len(pitches[timestep])):
            # if pitch is not active at this timestep
            if pitches[timestep][pitch] == 0:
                continue
            # pitch is active at this timestep and there is no pitch in pitch_dict
            if pitch not in pitch_dict:
                pitch_dict[pitch] = [{'start': timestep, 'end': timestep + 1}]
            # pitch is active and there is already a pitch present
            else:
                # pitch is already started, but did not end yet
                if pitch_dict[pitch][-1]['end'] == timestep:
                    pitch_dict[pitch][-1]['end'] = timestep + 1
                # there is a new pitch
                else:
                    pitch_dict[pitch].append({'start': timestep, 'end': timestep + 1})

    for pitch, pitch_list in pitch_dict.items():
        for timestep in pitch_list:
            note = pretty_midi.Note(velocity=50, pitch=pitch + 21, start=timestep['start'] * 0.04,
                                    end=timestep['end'] * 0.04)
            piano.notes.append(note)

    file.instruments.append(piano)
    file.write(path)
    return file


def save_midi_pianoroll(save_path, audio_path, midi_data: PrettyMIDI):
    # Plot piano roll based on midi
    plt.figure(figsize=(12, 6))
    plot_piano_roll(midi_data, 40, 90)
    plt.xlim([0, 30])
    plt.savefig(os.path.join(save_path, os.path.basename(audio_path) + '.pianoroll.pred.png'))


def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))
