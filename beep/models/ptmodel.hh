#pragma once

#include <string>
#include <torch/script.h>

#include "model.hh"
#include "third_party/eigen3/Eigen/Core"

class PyTorchModel : public Model
{
public:
  PyTorchModel( const std::string& path );

  void infer( span_view<float> true_audio,
              span_view<float> pred_audio,
              const size_t true_audio_last_timestamp,
              span<float> output ) override;

private:
  template<class T>
  using EigenMatrix = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  torch::jit::script::Module model_;

  EigenMatrix<float> bias_u, bias_r, bias_e;

  struct
  {
    EigenMatrix<float> weight;
    EigenMatrix<float> bias;
  } O1, O2, O3, O4, R_u, R_r, R_e, I_coarse, I_fine;
};
