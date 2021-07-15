#pragma once

#include "model.hh"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <thread>

#include "beep/models/test_model.h"
#include "beep/util/timer.hh"

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

class TensorFlowModel : public Model
{
public:
  TensorFlowModel() { model_.set_thread_pool( &eigen_dev_ ); }

  void infer( span_view<float> true_audio,
              span_view<float> pred_audio,
              const size_t true_audio_last_timestamp,
              span<float> output ) override;

private:
  Eigen::ThreadPool eigen_tp_ { static_cast<int>(
    std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 1 ) };
  Eigen::ThreadPoolDevice eigen_dev_ { &eigen_tp_, eigen_tp_.NumThreads() };
  TestModel model_ {};

  struct
  {
    std::unique_ptr<float[]> arg0_buffer;
  } model_buffers_ { std::make_unique<float[]>( model_.arg0_count() ) };
};
