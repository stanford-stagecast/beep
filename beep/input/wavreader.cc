#include "wavreader.hh"

#include <iostream>
#include <stdexcept>

using namespace std;

WavReader::WavReader( const string& path )
  : handle_( path, SFM_READ )
{
  /* make sure it's a wav file */
  if ( !( handle_.format() & SF_FORMAT_WAV ) ) {
    throw runtime_error( "not a WAV file: " + path );
  }
}

size_t WavReader::read( SafeEndlessBuffer<float>& buffer )
{
  auto writing_region = buffer.region( buffer.range_begin(), buffer.range_end() - buffer.range_begin() );
  return handle_.read( writing_region.begin(), writing_region.size() );
}
