#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "beep/audio/alsa_devices.hh"
#include "beep/input/wavreader.hh"
#include "beep/models/test_model.h"

using namespace std;

class Beep
{
public:
  Beep( const std::string& input );
  void play();

private:
  WavReader wav_reader_;

  // audio interface
  AudioInterface audio_output_ { "default", "audio output", SND_PCM_STREAM_PLAYBACK };
  ChannelPair audio_buffer_ { 32768 };
  size_t write_cursor_ { 0 };

  // neural net
  Eigen::ThreadPool eigen_tp_ { 2 };
  Eigen::ThreadPoolDevice eigen_dev_ { &eigen_tp_, eigen_tp_.NumThreads() };
  TestModel model_ {};
};

Beep::Beep( const string& input )
  : wav_reader_( input )
{
  audio_output_.initialize();
  audio_output_.start();
}

void Beep::play()
{
  bool eof = false;

  while ( not eof ) {
    while ( write_cursor_ < audio_buffer_.range_end() ) {
      const auto read_count = wav_reader_.read( audio_buffer_.ch1() );

      if ( read_count == 0 ) {
        eof = true;
        break;
      }

      write_cursor_ += read_count;
    }

    audio_output_.play( audio_buffer_, write_cursor_ );
    audio_buffer_.pop_before( audio_output_.cursor() );
  }
}

void usage( char* argv0 )
{
  cerr << "usage: " << argv0 << " WAVFILE" << endl;
}

int main( int argc, char* argv[] )
{
  try {
    if ( argc <= 0 ) {
      abort();
    }

    if ( argc != 2 ) {
      usage( argv[0] );
      return EXIT_FAILURE;
    }

    Beep beep { argv[1] };
    beep.play();

    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
