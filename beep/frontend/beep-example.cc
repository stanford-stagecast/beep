#include "beep/audio/alsa_devices.hh"

#include <cstdlib>
#include <iostream>

using namespace std;

void program_body()
{
  AudioInterface audio_output { "default", "audio output", SND_PCM_STREAM_PLAYBACK };
  audio_output.initialize();
  audio_output.start();

  ChannelPair fakeaudio { 32768 };
  size_t write_cursor = 0;

  while ( true ) {
    while ( write_cursor < fakeaudio.range_end() ) {
      fakeaudio.ch1().at( write_cursor ) = sin( 440.0 * 2 * M_PI * write_cursor / 48000.0 );
      fakeaudio.ch2().at( write_cursor ) = sin( 660.0 * 2 * M_PI * write_cursor / 48000.0 );
      write_cursor++;
    }

    audio_output.play( fakeaudio, write_cursor );

    fakeaudio.pop_before( audio_output.cursor() );
  }
}

int main()
{
  try {
    program_body();
    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
