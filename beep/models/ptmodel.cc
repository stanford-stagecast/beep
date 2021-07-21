#include "ptmodel.hh"

#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>

using namespace std;

/*
struct WaveRNN : torch::nn::Module
{
  WaveRNN( const size_t hidden_size = 896, const size_t quantisation = 256 )
  {
    using namespace torch;

    const size_t split_size = hidden_size / 2;

    R = register_module( "R", nn::Linear( nn::LinearOptions( hidden_size, 3 * hidden_size ).bias( false ) ) );
    O1 = register_module( "O1", nn::Linear( split_size, split_size ) );
    O2 = register_module( "O2", nn::Linear( split_size, quantisation ) );
    O3 = register_module( "O3", nn::Linear( split_size, split_size ) );
    O4 = register_module( "O4", nn::Linear( split_size, quantisation ) );
    I_coarse = register_module( "I_coarse", nn::Linear( nn::LinearOptions( 2, 3 * split_size ).bias( false ) ) );
    I_coarse = register_module( "I_fine", nn::Linear( nn::LinearOptions( 3, 3 * split_size ).bias( false ) ) );
    bias_u = register_parameter( "bias_u", torch::zeros( hidden_size ) );
    bias_r = register_parameter( "bias_r", torch::zeros( hidden_size ) );
    bias_e = register_parameter( "bias_e", torch::zeros( hidden_size ) );
  }

  torch::nn::Linear R { nullptr };
  torch::nn::Linear O1 { nullptr }, O2 { nullptr }, O3 { nullptr }, O4 { nullptr };
  torch::nn::Linear I_coarse { nullptr }, I_fine { nullptr };
  torch::Tensor bias_u, bias_r, bias_e;
};
*/

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> torch_to_eigen_2d( const at::Tensor& t )
{
  assert( t.dim() == 2 );
  using T = typename Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<T> result { t.data_ptr<float>(), t.size( 0 ), t.size( 1 ) };
  return result;
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> torch_to_eigen_1d( const at::Tensor& t )
{
  assert( t.dim() == 1 );
  using T = typename Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<T> result { t.data_ptr<float>(), t.size( 0 ), 1 };
  return result;
}

#define COPY_1D( x )                                                                                               \
  if ( param.name == #x ) {                                                                                        \
    x = torch_to_eigen_1d( param.value );                                                                          \
    continue;                                                                                                      \
  }

#define COPY_2D( x )                                                                                               \
  if ( param.name == #x ) {                                                                                        \
    x = torch_to_eigen_2d( param.value );                                                                          \
    continue;                                                                                                      \
  }

PyTorchModel::PyTorchModel( const string& path )
  : model_( torch::jit::load( path ) )
{
  model_.eval();

  for ( const auto& param : model_.named_parameters() ) {
    COPY_1D( bias_u );
    COPY_1D( bias_r );
    COPY_1D( bias_e );

    COPY_2D( R_u.weight );
    COPY_2D( R_r.weight );
    COPY_2D( R_e.weight );

    COPY_2D( O1.weight );
    COPY_2D( O2.weight );
    COPY_2D( O3.weight );
    COPY_2D( O4.weight );

    COPY_1D( O1.bias );
    COPY_1D( O2.bias );
    COPY_1D( O3.bias );
    COPY_1D( O4.bias );

    COPY_2D( I_coarse.weight );
    COPY_2D( I_fine.weight );
  }
}

void PyTorchModel::infer( span_view<float> true_audio,
                          span_view<float> pred_audio,
                          const size_t true_audio_last_timestamp,
                          span<float> output )
{
  // std::vector<torch::jit::IValue> inputs;
  // inputs.push_back( torch::zeros( { 2048 } ) );
  // copy( max( true_audio.begin(), true_audio.end() - 2048 ),
  //       true_audio.end(),
  //       static_cast<float*>( inputs.back().toTensor().data_ptr() ) );

  // auto o = module_.forward( inputs ).toTensor();
  // cout << o.size( 0 ) << endl;
}
