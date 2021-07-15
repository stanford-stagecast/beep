#pragma once

#include <torch/script.h>
#include <string>

#include "model.hh"

class PyTorchTestModel : public Model
{
public:
  PyTorchTestModel( const std::string& path )
    : module_( path )
  {}

private:
  torch::jit::script::Module module_;
};
