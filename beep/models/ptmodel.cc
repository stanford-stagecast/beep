#include "ptmodel.hh"

#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include <random>


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

#define RELU( x )                                                                                               \
  x.array().cwiseMax(0.).matrix();

#define SIGMOID( x )                                                                                               \
  (1. / (1. + (-x.array()).exp())).matrix();

#define TANH( x )                                                                                               \
  x.array().tanh().matrix();

// void SOFTMAX(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& input, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& output) {
//   // using arrays allows to call native vectorizable exp and log
//   Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> wMinusMax = input.rowwise() - input.colwise().maxCoeff();
//   output = (wMinusMax.rowwise() - wMinusMax.exp().colwise().sum().log()).exp();
//   //output = input;
// }

void combine_signal(const uint8_t* coarse, const uint8_t* fine, span<float> out) {
  for(int i = 0; i < 120; i++) {
    out[i] = (float)(coarse[i] * 256 + fine[i] - 1 << 15);
  }
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
  int hidden_size = 200;
  int split_size = hidden_size/2;
  EigenMatrix<float> b_coarse_u = bias_u.block(0,0,split_size,1);
  EigenMatrix<float> b_fine_u = bias_u.block(split_size,0,split_size,1);
  EigenMatrix<float> b_coarse_r = bias_r.block(0,0,split_size,1);
  EigenMatrix<float> b_fine_r = bias_r.block(split_size,0,split_size,1);
  EigenMatrix<float> b_coarse_e = bias_e.block(0,0,split_size,1);
  EigenMatrix<float> b_fine_e = bias_e.block(split_size,0,split_size,1);

  EigenMatrix<float> hidden = EigenMatrix<float>::Zero(hidden_size, 1);

  EigenMatrix<float> /*out_coarse_f, out_fine_f, */prev_output, fine_input;
  EigenMatrix<float> coarse_input_proj, fine_input_proj;
  EigenMatrix<float> I_coarse_u, I_coarse_r, I_coarse_e, I_fine_u, I_fine_r, I_fine_e;
  EigenMatrix<float> R_coarse_u, R_coarse_r, R_coarse_e, R_fine_u, R_fine_r, R_fine_e;

  EigenMatrix<float> R_u_out, R_r_out, R_e_out;
  EigenMatrix<float> u, r, e, hidden_coarse, hidden_fine;

  uint8_t out_coarse = 0;
  uint8_t out_fine = 0;
  uint8_t coarse_outputs[120];
  uint8_t fine_outputs[120];
  float out_coarse_f, out_fine_f;
  std::random_device rd;
  std::mt19937 rng(rd());

  for (int i = 0; i < 120; i++) {
    // Split into two hidden states
    EigenMatrix<float> hidden_coarse = hidden.block(0,0,split_size,1);
    EigenMatrix<float> hidden_fine = hidden.block(split_size,0,split_size,1);

    // Scale and concat previous predictions
    out_coarse_f = /* EigenMatrix<float> */((float)out_coarse / 127.5 - 1);
    out_fine_f = /*EigenMatrix<float>*/((float)out_fine / 127.5 - 1);
    prev_output << out_coarse_f, out_fine_f;

    // Project input 
    coarse_input_proj = prev_output * I_coarse.weight.transpose();
    I_coarse_u = coarse_input_proj.block(0,0,split_size,1);
    I_coarse_r = coarse_input_proj.block(split_size,0,split_size,1);
    I_coarse_e = coarse_input_proj.block(2*split_size,0,split_size,1);

    R_u_out = hidden * R_u.weight.transpose();
    R_r_out = hidden * R_r.weight.transpose();
    R_e_out = hidden * R_e.weight.transpose();

    R_coarse_u = R_u_out.block(0,0,split_size,1);
    R_coarse_r = R_r_out.block(0,0,split_size,1);
    R_coarse_e = R_e_out.block(0,0,split_size,1);

    R_fine_u = R_u_out.block(split_size,0,split_size,1);
    R_fine_r = R_r_out.block(split_size,0,split_size,1);
    R_fine_e = R_e_out.block(split_size,0,split_size,1);

    // Compute the coarse gates
    // u = SIGMOID(R_coarse_u + I_coarse_u + b_coarse_u);
    //r = SIGMOID(R_coarse_r + I_coarse_r + b_coarse_r);
    //e = TANH(r * R_coarse_e + I_coarse_e + b_coarse_e);
    //hidden_coarse = u * hidden_coarse + (1. - u) * e;

    // // Compute the coarse output
    // out_coarse = RELU(hidden_coarse * O1.weight.transpose() + O1.bias); // * O2.weight.transpose() + O2.bias;
    // SOFTMAX(out_coarse, posterior);
    
    // std::discrete_distribution<int> dist1(posterior.begin(), posterior.end());
    // out_coarse = dist1(rng);

    // // Project the [prev outputs and predicted coarse sample]
    // coarse_pred = EigenMatrix<float>(out_coarse)/ 127.5 - 1;
    // fine_input << prev_outputs, coarse_pred;
    // fine_input_proj = fine_input * I_fine.weight.transpose();

    // I_fine_u = fine_input_proj.block(0,0,split_size,1);
    // I_fine_r = fine_input_proj.block(split_size,0,split_size,1);
    // I_fine_e = fine_input_proj.block(2*split_size,0,split_size,1);

    // // Compute the fine gates
    // u = SIGMOID(R_fine_u + I_fine_u + b_fine_u);
    // r = SIGMOID(R_fine_r + I_fine_r + b_fine_r);
    // e = TANH(r * R_fine_e + I_fine_e + b_fine_e);
    // hidden_fine = u * hidden_fine + (1. - u) * e;

    // // Compute the fine output
    // out_fine = RELU(hidden_fine * O1.weight.transpose() + O1.bias); // * O2.weight.transpose() + O2.bias;
    // SOFTMAX(out_fine, posterior);

    // std::discrete_distribution<int> dist2(posterior.begin(), posterior.end());
    // out_fine = dist2(rng);

    // coarse_outputs[i] = out_coarse;
    // fine_outputs[i] = out_fine;

    // hidden << hidden_coarse, hidden_fine;

  }
  combine_signal(coarse_outputs, fine_outputs, output);
}
