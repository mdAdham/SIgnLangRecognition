from infrastructure.midi.Midi import Midi

# we can use e “CC#1–#31, CC#64–#95”  to set different functions.
class rt505:
    def __init__(self, midi: Midi):
        self.midi = midi

    def play(self):
        self.midi.send_message([0xFA])
    def pause(self):
        self.midi.send_message([0xFC])
    def select_bank(self, number: int):
        #todo