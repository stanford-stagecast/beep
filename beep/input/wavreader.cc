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

size_t WavReader::read( span<float> out_buffer )
{
  return handle_.read( out_buffer.begin(), out_buffer.size() );
}
