#pragma once

#include <torch/script.h>
#include <string>

#include "model.hh"

class PyTorchModel : public Model
{
public:
  PyTorchModel( const std::string& path )
    : module_( path )
  {}

  void infer( span_view<float> true_audio,
                      span_view<float> pred_audio,
                      const size_t true_audio_last_timestamp,
                      span<float> output ) override;

private:
  torch::jit::script::Module module_;
};
