#pragma once

#include <sndfile.hh>

#include "beep/util/spans.hh"

class WavReader
{
private:
  SndfileHandle handle_;

public:
  WavReader( const std::string& path );

  size_t sample_rate() const { return handle_.samplerate(); }
  size_t channels() const { return handle_.channels(); }
  size_t frame_count() const { return handle_.frames(); }

  size_t read( span<float> out_buffer );
};
