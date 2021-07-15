#include "tfmodel.hh"

using namespace std;

void TensorFlowModel::infer( span_view<float> true_audio,
                             [[maybe_unused]] span_view<float> pred_audio,
                             [[maybe_unused]] const size_t true_audio_last_timestamp,
                             span<float> output )
{
  GlobalScopeTimer<Timer::Category::Inference> _;

  if ( model_.result0_count() < 0 || output.size() != static_cast<size_t>( model_.result0_count() ) ) {
    throw runtime_error( "infer: expected an output of size "s + to_string( model_.result0_count() ) + ", got "s
                         + to_string( output.size() ) + " instead"s );
  }

  fill_n( model_buffers_.arg0_buffer.get(), model_.arg0_count(), 0.f );
  copy( max( true_audio.begin(), true_audio.end() - model_.arg0_count() ),
        true_audio.end(),
        model_buffers_.arg0_buffer.get() );

  model_.set_arg0_data( model_buffers_.arg0_buffer.get() );
  model_.Run();

  copy( model_.result0_data(), model_.result0_data() + model_.result0_count(), output.begin() );
}
