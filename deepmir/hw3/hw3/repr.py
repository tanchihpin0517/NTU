from functools import cache
from itertools import chain
from pathlib import Path
import miditoolkit
import collections
import numpy as np
import json

# from . import utils

WORD_SIZE = 3
DEFAULT_SUBBEAT_RANGE = np.arange(0, 16, dtype=int)
DEFAULT_PIANO_RANGE = np.arange(21, 109, dtype=int)
DEFAULT_VELOCITY_BINS = np.linspace(0,  124, 31+1, dtype=int) # midi velocity: 0~127
DEFAULT_BPM_BINS = np.linspace(32, 224, 64+1, dtype=int)
DEFAULT_DURATION_RANGE = np.arange(1, 1+32, dtype=int)
DEFAULT_CHORD_ROOTS = [
    "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#",
]
DEFAULT_CHORD_QUALITY = [
    "+",
    "/o7",
    "7",
    "M",
    "M7",
    "m",
    "m7",
    "o",
    "o7",
    "sus2",
    "sus4",
]

# class Event:
#     PAD = "pad"
#     IGN = "ign"
#
#     @classmethod
#     def from_list(cls):
#         raise NotImplementedError
#
#     def to_list(self):
#         raise NotImplementedError
#
# class PadEvent(Event):
#     NUM_CLASSES = 1
#
#     @staticmethod
#     @cache
#     def defined():
#         return [Event.PAD]
#
#     def __init__(self):
#         raise NotImplementedError
#
# class SpecEvent(Event):
#     NUM_CLASSES = 1
#
#     @staticmethod
#     @cache
#     def defined():
#         defs = [f'spec_{Event.PAD}', 'bos', 'eos', 'unk', 'ss', 'se']
#         return defs
#
#     @classmethod
#     def from_list(cls, inp):
#         return SpecEvent(inp[0])
#
#     def to_list(self):
#         out = [f'spec_{Event.PAD}'] * WORD_SIZE
#         out[0] = self.name
#         return out
#
#     def __init__(self, name):
#         assert name in SpecEvent.defined()
#         self.name = name
#
#     def __repr__(self):
#         return f'Spec({self.name})'
#
# class BarEvent(Event):
#     NUM_CLASSES = 1
#
#     @staticmethod
#     @cache
#     def defined():
#         defs = [f'bar_{Event.PAD}', 'bar']
#         return defs
#
#     @classmethod
#     def from_list(cls, bar_num=None):
#         return BarEvent(bar_num if bar_num is not None else -1)
#
#     def to_list(self):
#         out = [f'bar_{Event.PAD}'] * WORD_SIZE
#         out[0] = 'bar'
#         return out
#
#     def __init__(self, time):
#         self.time = time
#
#     def __repr__(self):
#         return f'Bar({self.time})'
#
# class MetricEvent(Event):
#     NUM_CLASSES = 3
#
#     @staticmethod
#     @cache
#     def defined():
#         defs = [f'metric_{Event.PAD}']
#
#         for i in DEFAULT_SUBBEAT_RANGE:
#             defs.append(f'subbeat_{i}')
#
#         defs.append(f'tempo_{Event.IGN}')
#         for tempo in DEFAULT_BPM_BINS:
#             defs.append(f'tempo_{tempo}')
#
#         defs.append(f'chord_{Event.IGN}')
#         defs.append('chord_N_N')
#         for root in DEFAULT_CHORD_ROOTS:
#             for quality in DEFAULT_CHORD_QUALITY:
#                 defs.append(f"chord_{root}_{quality}")
#
#         return defs
#
#     @classmethod
#     def from_list(cls, inp):
#         vals = [ i.split('_', 1)[-1] for i in inp[:cls.NUM_CLASSES] ]
#         if vals[2] != 'ign':
#             vals[2] = int(vals[2])
#         return MetricEvent(int(vals[0]), vals[1], vals[2])
#
#     def to_list(self):
#         out = [f'metric_{Event.PAD}'] * WORD_SIZE
#         out[0] = f'subbeat_{self.subbeat}'
#         out[1] = f'chord_{self.chord}'
#         out[2] = f'tempo_{self.tempo}'
#         return out
#
#
#     def __init__(self, subbeat, chord, tempo):
#         self.subbeat = subbeat
#         self.chord = chord
#         self.tempo = tempo
#
#     @property
#     def tempo(self):
#         return self._tempo
#
#     @tempo.setter
#     def tempo(self, tempo):
#         if tempo == 'ign':
#             self._tempo = tempo
#         else:
#             self._tempo = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS-tempo))]
#
#     def __repr__(self):
#         return f'Metric({self.subbeat}, chord={self.chord}, tempo={self.tempo})'
#
# class NoteEvent(Event):
#     NUM_CLASSES = 3
#
#     @staticmethod
#     @cache
#     def defined():
#         defs = [f'note_{Event.PAD}']
#
#         for pitch in DEFAULT_PIANO_RANGE:
#             defs.append(f'pitch_{pitch}')
#
#         for velocity in DEFAULT_VELOCITY_BINS:
#             defs.append(f'velocity_{velocity}')
#
#         for duration in DEFAULT_DURATION_RANGE:
#             defs.append(f'duration_{duration}')
#
#         return defs
#
#     @classmethod
#     def from_list(cls, inp):
#         vals = [ i.split('_', 1)[-1] for i in inp[:cls.NUM_CLASSES] ]
#         return NoteEvent(int(vals[0]), int(vals[1]), int(vals[2]))
#
#     def to_list(self):
#         out = [f'note_{Event.PAD}'] * WORD_SIZE
#         out[0] = f'pitch_{self.pitch}'
#         out[1] = f'duration_{self.duration}'
#         out[2] = f'velocity_{self.velocity}'
#         return out
#
#     def __init__(self, pitch, duration, velocity):
#         self.pitch = pitch
#         self.duration = duration
#         self.velocity = velocity
#
#     @property
#     def duration(self):
#         return min(self._duration, max(DEFAULT_DURATION_RANGE))
#
#     @duration.setter
#     def duration(self, duration):
#         self._duration = duration
#
#     @property
#     def velocity(self):
#         return self._velocity
#
#     @velocity.setter
#     def velocity(self, velocity):
#         self._velocity = DEFAULT_VELOCITY_BINS[np.argmin(abs(DEFAULT_VELOCITY_BINS-velocity))]
#
#     def __repr__(self):
#         return f'Note(pitch={self.pitch}, dur={self.duration}, vel={self.velocity})'

class Tokenizer:
    def __init__(self, vocab_file, beat_div = 4, ticks_per_beat = 480):
        self.vocab = Vocab(vocab_file)
        self.beat_div = beat_div
        self.ticks_per_beat = ticks_per_beat

    def get_song_from_midi_file(self, midi_file):
        midi = miditoolkit.midi.parser.MidiFile(midi_file)
        return self.get_song_from_midi(midi)

    def get_song_from_midi(self, midi):
        beat_per_bar = 4
        song = midi_to_song(midi, beat_per_bar, self.beat_div)
        song['events'] = ['spec_ss'] + get_events(song) + ['spec_se']
        song['ids'] = list(map(self.e2i, song['events']))
        return song

    def events_to_midi(self, events, beat_per_bar):
        midi = events_to_midi(events, beat_per_bar, self.ticks_per_beat, self.beat_div)
        return midi

    def e2i(self, e):
        return self.vocab.e2i[e]

    def i2e(self, eid):
        return self.vocab.i2e[eid]

class Vocab:
    def __init__(self, vocab_file):
        self.vocab = json.load(open(vocab_file))
        self.e2i = {}
        self.i2e = []
        count = 0
        for _, words in self.vocab.items():
            self.i2e.extend(words)
            for word in words:
                self.e2i[word] = count
                count += 1

    @staticmethod
    def gen_vocab(file_path):
        vocab = Vocab._gen_vocab()
        json.dump(vocab, open(file_path.with_suffix(".json"), 'w'), indent=4)

    @staticmethod
    def _gen_vocab():
        spec = ['spec_pad', 'spec_bos', 'spec_eos', 'spec_unk', 'spec_ss', 'spec_se']
        bar = ['bar']
        beat = [f'subbeat_{i}' for i in DEFAULT_SUBBEAT_RANGE]
        tempo = [f'tempo_{i}' for i in DEFAULT_BPM_BINS]
        chord = ['chord_N_N']
        for root in DEFAULT_CHORD_ROOTS:
            for quality in DEFAULT_CHORD_QUALITY:
                chord.append(f"chord_{root}_{quality}")
        pitch = [f'pitch_{i}' for i in DEFAULT_PIANO_RANGE]
        duration = [f'duration_{i}' for i in DEFAULT_DURATION_RANGE]
        velocity = [f'velocity_{i}' for i in DEFAULT_VELOCITY_BINS]
        vocab = {
            'spec': spec,
            'bar': bar,
            'beat': beat,
            'tempo': tempo,
            'chord': chord,
            'pitch': pitch,
            'duration': duration,
            'velocity': velocity,
        }
        return vocab

    def __len__(self):
        return len(self.i2e)

    def __repr__(self):
        out = []
        for t, words in self.vocab.items():
            out.append(f"type: {t}")
            for word in words:
                out.append(f"\t{word}")
        return '\n'.join(out)

def midi_to_song(midi_obj, beat_per_bar, beat_div):
    assert midi_obj.ticks_per_beat % beat_div == 0
    grid_resol = midi_obj.ticks_per_beat // beat_div

    # load notes
    instr_notes = collections.defaultdict(list)
    for instr in midi_obj.instruments:
        for note in instr.notes:
            instr_notes[instr.name].append(note)
        instr_notes[instr.name].sort(key=lambda x: x.start)

    # load chords
    chords = []
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] != 'global' and \
        'Boundary' not in marker.text.split('_')[0]:
            chords.append(marker)
    chords.sort(key=lambda x: x.time)

    # load tempos
    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: x.time)

    # load labels
    labels = []
    for marker in midi_obj.markers:
        if 'Boundary' in marker.text.split('_')[0]:
            labels.append(marker)
    labels.sort(key=lambda x: x.time)

    # load global bpm
    gobal_bpm = tempos[0].tempo
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] == 'global' and \
            marker.text.split('_')[1] == 'bpm':
            gobal_bpm = int(marker.text.split('_')[2])

    # process notes
    intsr_gird = dict()
    for key in instr_notes.keys():
        notes = instr_notes[key]
        note_grid = collections.defaultdict(list)
        for note in notes:
            # quantize start
            quant_time = round(note.start / grid_resol)

            # velocity
            # note.velocity = DEFAULT_VELOCITY_BINS[
            #     np.argmin(abs(DEFAULT_VELOCITY_BINS-note.velocity))]
            # note.velocity = max(MIN_VELOCITY, note.velocity)

            # offset of start
            # note.shift = note.start - quant_time 
            # note.shift = DEFAULT_SHIFT_BINS[np.argmin(abs(DEFAULT_SHIFT_BINS-note.shift))]

            # duration
            note_duration = note.end - note.start
            ntick_duration = round(note_duration / grid_resol)
            ntick_duration = max(ntick_duration, 1) # dur >= 1
            note.duration = ntick_duration

            # append
            note_grid[quant_time].append(note)

        # sort
        for time in note_grid.keys():
            note_grid[time].sort(key=lambda x: -x.pitch)

        # set to track
        intsr_gird[key] = note_grid.copy()

    # process chords
    chord_grid = collections.defaultdict(list)
    for chord in chords:
        quant_time = round(chord.time / grid_resol)
        chord_grid[quant_time] = [chord] # NOTE: only one chord per time

    # process tempo
    tempo_grid = collections.defaultdict(list)
    for tempo in tempos:
        quant_time = round(tempo.time / grid_resol)
        # tempo.tempo = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS-tempo.tempo))]
        tempo_grid[quant_time] = [tempo] # NOTE: only one tempo per time

    all_bpm = [tempo[0].tempo for _, tempo in tempo_grid.items()]
    assert len(all_bpm) > 0, ' No tempo changes in midi file.'
    average_bpm = sum(all_bpm) / len(all_bpm)

    # process boundary
    label_grid = collections.defaultdict(list)
    for label in labels:
        quant_time = round(label.time / grid_resol)
        label_grid[quant_time] = [label]

    # process global bpm
    # gobal_bpm = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS-gobal_bpm))]

    # collect
    song = {
        'tokens': [],
        'ids': [],
        'data': {
            'notes': intsr_gird,
            'chords': chord_grid,
            'tempos': tempo_grid,
            'labels': label_grid,
        },
        'metadata': {
            'global_bpm': gobal_bpm,
            'average_bpm': average_bpm,
            'beat_per_bar': beat_per_bar,
            'beat_div': beat_div,
        },
    }

    return song

def get_events(song):
    beat_per_bar = song['metadata']['beat_per_bar']
    beat_div = song['metadata']['beat_div']

    events = []

    max_grid = list(chain(song['data']['tempos'].keys(), song['data']['chords'].keys()))
    for _, v in song['data']['notes'].items():
        max_grid.extend(v.keys())
    max_grid = max(max_grid)

    grid_per_bar = beat_per_bar * beat_div
    for beat_i in range(0, max_grid + 1, grid_per_bar):
        events.append('bar')
        for i in range(beat_i, min(beat_i + grid_per_bar, max_grid + 1)):
            tmp = []
            if i in song['data']['chords']:
                for chord in song['data']['chords'][i]:
                    chord = chord.text.rsplit('_', 1)[0]
                    tmp.append(f'chord_{chord}')
            if i in song['data']['tempos']:
                for tempo in song['data']['tempos'][i]:
                    tempo = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS-tempo.tempo))]
                    tmp.append(f'tempo_{tempo}')
            for _, instr in song['data']['notes'].items():
                if i in instr:
                    for note in instr[i]:
                        velocity = DEFAULT_VELOCITY_BINS[np.argmin(abs(DEFAULT_VELOCITY_BINS-note.velocity))]
                        duration = DEFAULT_DURATION_RANGE[np.argmin(abs(DEFAULT_DURATION_RANGE-note.duration))]
                        tmp.append(f'pitch_{note.pitch}')
                        tmp.append(f'duration_{duration}')
                        tmp.append(f'velocity_{velocity}')
            if len(tmp) != 0:
                events.append(f'subbeat_{i-beat_i}')
                events.extend(tmp)

    return events

def events_to_midi(events, beat_per_bar, ticks_per_beat, beat_div):
    assert ticks_per_beat % beat_div == 0
    ticks_per_subbeat = ticks_per_beat // beat_div
    ticks_per_bar = ticks_per_beat * beat_per_bar

    midi = miditoolkit.midi.parser.MidiFile()
    midi.ticks_per_beat = ticks_per_beat
    track = miditoolkit.midi.containers.Instrument(program=0, is_drum=False, name='piano')
    midi.instruments = [track]

    bar_tick = 0
    subbeat_tick = 0
    first_bar = True

    pitch, velocity, duration = 0, 0, 0

    for event in events:
        if event.startswith('spec'):
            pass
        elif event.startswith('bar'):
            if not first_bar:
                bar_tick += ticks_per_bar
            first_bar = False

        elif event.startswith('subbeat'):
            subbeat = int(event.split('_')[1])
            subbeat_tick = subbeat * ticks_per_subbeat

        elif event.startswith('chord'):
            pass

        elif event.startswith('tempo'):
            tempo = int(event.split('_')[1])
            m = miditoolkit.midi.containers.TempoChange(time=bar_tick+subbeat_tick, tempo=tempo)
            midi.tempo_changes.append(m)

        elif event.startswith('pitch'):
            pitch = int(event.split('_')[1])

        elif event.startswith('duration'):
            duration = int(event.split('_')[1])
            duration = duration * ticks_per_subbeat

        elif event.startswith('velocity'):
            velocity = int(event.split('_')[1])
            n = miditoolkit.Note(
                start=bar_tick+subbeat_tick,
                end=bar_tick+subbeat_tick+duration,
                pitch=pitch,
                velocity=velocity
            )
            midi.instruments[0].notes.append(n)

        else:
            raise ValueError(f'Unknown event: {event}')

    return midi

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gen_vocab', action='store_true')
    subparsers = parser.add_subparsers(dest="command")
    cmd_gen_vocab = subparsers.add_parser('gen_vocab')
    cmd_gen_vocab.add_argument('--output_file', type=Path, required=True)
    cmd_test = subparsers.add_parser('test')
    cmd_test.add_argument('--vocab_file', type=Path, required=True)
    cmd_test.add_argument('--midi_file', type=Path, required=True)
    cmd_test.add_argument('--output_file', type=Path, required=True)
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        exit()
    elif args.command == 'gen_vocab':
        Vocab.gen_vocab(args.output_file)
        vocab = Vocab(args.output_file)
        print(vocab)
        print("vocab size:", len(vocab))
    elif args.command == 'test':
        tokenizer = Tokenizer(args.vocab_file)
        print(tokenizer.vocab)
        song = tokenizer.get_song_from_midi_file(args.midi_file)
        print(song['events'])
        print(song['ids'])
        midi = tokenizer.events_to_midi(song['events'], 4)
        midi.dump(args.output_file)

    # if args.gen_vocab:
    #     vocab = Vocab.gen_vocab()
    #     print(*vocab, sep='\n')
    # else:
    #     midi = miditoolkit.midi.parser.MidiFile('0.mid')
    #     print(midi.ticks_per_beat)
    #     midi = miditoolkit.midi.parser.MidiFile('1.mid')
    #     print(midi.ticks_per_beat)
    #     song = midi_to_song(midi, 4)
    #     events = song_to_events(song)
    #     print("\n".join(map(str, events)))
    #     midi = events_to_midi(events, 480, 4)
    #     midi.dump('out.mid')
    #     print(Vocab.gen_vocab())
