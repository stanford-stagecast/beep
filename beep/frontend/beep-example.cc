#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "beep/audio/alsa_devices.hh"
#include "beep/input/wavreader.hh"
#include "beep/util/timer.hh"

#include "beep/models/ptmodel.hh"
#include "beep/models/tfmodel.hh"

using namespace std;

constexpr size_t operator"" _samples( unsigned long long int n )
{
  return n;
}

class Beep
{
public:
  Beep( const std::string& input, const string& model_type, const string& model_path = {} );
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

  unique_ptr<Model> model_;
};

Beep::Beep( const string& input, const string& model_type, const string& model_path )
  : sample_count_( WavReader { input }.frame_count() )
  , true_audio_( sample_count_ )
  , predicted_audio_( sample_count_, 0.f )
{
  if ( model_type == "torch" ) {
    model_ = make_unique<PyTorchModel>( model_path );
  } else if ( model_type == "tensorflow" ) {
    model_ = make_unique<TensorFlowModel>();
  } else {
    throw runtime_error( "unknown model type: " + model_type );
  }

  WavReader wav_reader { input };
  if ( wav_reader.read( { true_audio_.data(), true_audio_.size() } ) != sample_count_ ) {
    throw runtime_error( "not all the samples were read" );
  }
}

void Beep::predict()
{
  constexpr size_t interval = 120_samples; /* 2.5ms */
  constexpr size_t latency = 1440_samples; /* true audio is 30ms behind */

  const size_t last_interval_start = interval * ( sample_count_ / interval - 1 );

  // can't do much about the first interval, since no real audio is available
  fill_n( predicted_audio_.data(), interval, 0 );

  auto print_status = [&]( const auto abs_time ) {
    static auto last_status_print = chrono::steady_clock::now();
    const auto now = chrono::steady_clock::now();

    if ( now - last_status_print >= chrono::seconds { 1 } ) {
      last_status_print = now;
      cout << "Inference: " << setprecision( 1 ) << fixed
           << ( abs_time / interval / ( 1.0 * last_interval_start / interval ) * 100 )
           << "% (abs time = " << abs_time << ")" << endl;
    }
  };

  for ( size_t abs_time = 0; abs_time < last_interval_start; abs_time += interval ) {
    print_status( abs_time );

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

    model_->infer( input_true_audio, input_pred_audio, input_true_audio_end, output_pred_audio );
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

void usage( char* argv0 )
{
  cerr << "usage: " << argv0 << " WAVFILE MODEL_TYPE=<torch|tensorflow> [MODEL_PATH]" << endl;
}

int main( int argc, char* argv[] )
{
  try {
    if ( argc <= 0 ) {
      abort();
    }

    if ( argc < 3 ) {
      usage( argv[0] );
      return EXIT_FAILURE;
    }

    string wav_path { argv[1] };
    string model_type { argv[2] };
    string model_path { argc > 3 ? argv[3] : "" };

    Beep beep { wav_path, model_type, model_path };
    beep.predict();
    beep.play();

    global_timer().summary( cout );

    return EXIT_SUCCESS;
  } catch ( const exception& e ) {
    cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
