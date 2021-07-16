#include "ptmodel.hh"

using namespace std;

void PyTorchModel::infer( span_view<float> true_audio,
                          span_view<float> pred_audio,
                          const size_t true_audio_last_timestamp,
                          span<float> output )
{}
