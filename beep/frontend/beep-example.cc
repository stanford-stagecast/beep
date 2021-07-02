#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "beep/audio/alsa_devices.hh"
#include "beep/models/test_model.h"

using namespace std;

void program_body()
{
  // taking care of the model
  Eigen::ThreadPool eigen_tp { 2 };
  Eigen::ThreadPoolDevice eigen_dev { &eigen_tp, eigen_tp.NumThreads() };

  TestModel model {};
  model.set_thread_pool( &eigen_dev );

  array<float, 10000> input;
  for ( size_t i = 0; i < input.size(); i++ ) {
    input[i] = 1.0 * i / input.size();
  }

  std::copy( input.data(), input.data() + input.size(), model.arg0_data() );
  model.Run();
  cout << "Result: " << model.result0(0, 0, 0) << endl;

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
