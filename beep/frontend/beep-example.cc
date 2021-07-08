#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "beep/audio/alsa_devices.hh"
#include "beep/input/wavreader.hh"
#include "beep/models/test_model.h"
#include "beep/util/timer.hh"

using namespace std;

constexpr size_t operator"" _samples( unsigned long long int n )
{
  return n;
}

class Beep
{
public:
  Beep( const std::string& input );
  void predict();
  void play();

private:
  void infer( span_view<float> true_audio,
              span_view<float> pred_audio,
              const size_t true_audio_last_timestamp,
              span<float> output );

  // true & predicted audio
  const size_t sample_count_;
  vector<float> true_audio_;
  vector<float> predicted_audio_;

  // neural net
  Eigen::ThreadPool eigen_tp_ { static_cast<int>(
    thread::hardware_concurrency() > 0 ? thread::hardware_concurrency() : 1 ) };
  Eigen::ThreadPoolDevice eigen_dev_ { &eigen_tp_, eigen_tp_.NumThreads() };
  TestModel model_ {};

  struct
  {
    unique_ptr<float[]> arg0_buffer;
  } model_buffers_ { make_unique<float[]>( model_.arg0_count() ) };
};

Beep::Beep( const string& input )
  : sample_count_( WavReader { input }.frame_count() )
  , true_audio_( sample_count_ )
  , predicted_audio_( sample_count_, 0.f )
{
  WavReader wav_reader { input };
  if ( wav_reader.read( { true_audio_.data(), true_audio_.size() } ) != sample_count_ ) {
    throw runtime_error( "not all the samples were read" );
  }

  model_.set_thread_pool( &eigen_dev_ );
}

void Beep::predict()
{
  constexpr size_t interval = 120_samples; /* 2.5ms */
  constexpr size_t latency = 1440_samples; /* true audio is 30ms behind */

  const size_t last_interval_start = interval * ( sample_count_ / interval - 1 );

  // can't do much about the first interval, since no real audio is available
  fill_n( predicted_audio_.data(), interval, 0 );

  auto last_status_print = chrono::steady_clock::now();

  for ( size_t abs_time = 0; abs_time < last_interval_start; abs_time += interval ) {

    const auto now = chrono::steady_clock::now();
    if ( now - last_status_print >= chrono::seconds { 1 } ) {
      last_status_print = now;
      cout << "Inference: " << setprecision( 1 ) << fixed
           << ( abs_time / interval / ( 1.0 * last_interval_start / interval ) * 100 )
           << "% (abs time = " << abs_time << ")" << endl;
    }

    // the goal is to predict `interval` samples of true audio for relative
    // time t=[interval, interval + interval)

    const size_t input_true_audio_end = abs_time > latency ? static_cast<size_t>( abs_time - latency ) : 0;
    const size_t input_pred_audio_end = abs_time + interval;

    const size_t output_pred_audio_start = input_pred_audio_end;
    const size_t output_pred_audio_end = min( output_pred_audio_start + interval, sample_count_ );

    // the inputs
    span_view<float> input_true_audio { true_audio_.data(), input_true_audio_end };
    span_view<float> input_pred_audio { predicted_audio_.data(), input_pred_audio_end };

    // the output
    span<float> output_pred_audio { predicted_audio_.data() + output_pred_audio_start,
                                    output_pred_audio_end - output_pred_audio_start };

    if ( abs_time == 0 ) {
      fill_n( output_pred_audio.begin(), output_pred_audio.size(), 0 );
      continue;
    }

    infer( input_true_audio, input_pred_audio, input_true_audio_end, output_pred_audio );
  }
}

void Beep::play()
{
  AudioInterface audio_output { "default", "audio output", SND_PCM_STREAM_PLAYBACK };
  ChannelPair playback_buffer { 32768 };
  size_t write_cursor { 0 };

  audio_output.initialize();
  audio_output.start();

  while ( write_cursor < sample_count_ ) {
    while ( write_cursor < playback_buffer.range_end() && write_cursor < sample_count_ ) {
      playback_buffer.ch1().at( write_cursor ) = true_audio_.at( write_cursor );
      playback_buffer.ch2().at( write_cursor ) = predicted_audio_.at( write_cursor );
      write_cursor++;
    }

    audio_output.play( playback_buffer, write_cursor );
    playback_buffer.pop_before( audio_output.cursor() );
  }
}

void Beep::infer( span_view<float> true_audio,
                  [[maybe_unused]] span_view<float> pred_audio,
                  [[maybe_unused]] const size_t true_audio_last_timestamp,
                  span<float> output )
{
  GlobalScopeTimer<Timer::Category::Inference> _;

  if ( model_.result0_count() < 0 || output.size() != static_cast<size_t>( model_.result0_count() ) ) {
    throw runtime_error( "Beep::infer: expected an output of size "s + to_string( model_.result0_count() )
                         + ", got "s + to_string( output.size() ) + " instead"s );
  }

  fill_n( model_buffers_.arg0_buffer.get(), model_.arg0_count(), 0.f );
  copy( max( true_audio.begin(), true_audio.end() - model_.arg0_count() ),
        true_audio.end(),
        model_buffers_.arg0_buffer.get() );

  model_.set_arg0_data( model_buffers_.arg0_buffer.get() );
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
    beep.predict();
    beep.play();

    global_timer().summary( cout );

    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
