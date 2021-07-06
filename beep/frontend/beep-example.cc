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
#include "beep/util/timer.hh"

using namespace std;

class Beep
{
public:
  Beep( const std::string& input );
  void play();

private:
  void infer( span_view<float> input, span<float> output );

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
  model_.set_thread_pool( &eigen_dev_ );
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

void Beep::infer( span_view<float> input, span<float> output )
{
  GlobalScopeTimer<Timer::Category::Inference> _;

  if ( model_.arg0_count() < 0 || input.size() != static_cast<size_t>( model_.arg0_count() ) ) {
    throw runtime_error( "Beep::infer: expected an input of size "s + to_string( model_.arg0_count() ) + ", got "s
                         + to_string( input.size() ) + " instead"s );
  }

  if ( model_.result0_count() < 0 || output.size() != static_cast<size_t>( model_.result0_count() ) ) {
    throw runtime_error( "Beep::infer: expected an output of size "s + to_string( model_.result0_count() )
                         + ", got "s + to_string( output.size() ) + " instead"s );
  }

  model_.set_arg0_data( input.begin() );
  model_.Run();

  copy( model_.result0_data(), model_.result0_data() + model_.result0_count(), output.begin() );
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

    global_timer().summary( cout );

    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
