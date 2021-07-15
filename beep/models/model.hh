#pragma once

#include "beep/util/spans.hh"

class Model
{
public:
  virtual void infer( span_view<float> true_audio,
                      span_view<float> pred_audio,
                      const size_t true_audio_last_timestamp,
                      span<float> output )
    = 0;
};
