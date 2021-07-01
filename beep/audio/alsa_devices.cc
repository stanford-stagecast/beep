#include <algorithm>
#include <alsa/asoundlib.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <tuple>

#include "alsa_devices.hh"
#include "beep/util/exception.hh"

using namespace std;
using namespace std::chrono;

class alsa_error_category : public error_category
{
public:
  const char* name() const noexcept override { return "alsa_error_category"; }
  string message( const int return_value ) const noexcept override { return snd_strerror( return_value ); }
};

class alsa_error : public system_error
{
  string what_;

public:
  alsa_error( const string& context, const int err )
    : system_error( err, alsa_error_category() )
    , what_( context + ": " + system_error::what() )
  {}

  const char* what() const noexcept override { return what_.c_str(); }
};

int alsa_check( const char* context, const int return_value )
{
  if ( return_value >= 0 ) {
    return return_value;
  }
  throw alsa_error( context, return_value );
}

int alsa_check( const string& context, const int return_value )
{
  return alsa_check( context.c_str(), return_value );
}

#define alsa_check_easy( expr ) alsa_check( #expr, expr )

AudioInterface::AudioInterface( const string_view interface_name,
                                const string_view annotation,
                                const snd_pcm_stream_t stream )
  : interface_name_( interface_name )
  , annotation_( annotation )
  , pcm_( [&] {
    snd_pcm_t* ret;
    alsa_check_easy( snd_pcm_open( &ret, interface_name_.c_str(), stream, 0 ) );
    notnull( "snd_pcm_open", ret );
    return ret;
  }() )
  , poll_fd_( get_poll_fd() )
{
  check_state( SND_PCM_STATE_OPEN );
}

PCMFD AudioInterface::get_poll_fd()
{
  const int count = alsa_check_easy( snd_pcm_poll_descriptors_count( pcm_ ) );
  if ( count < 1 or count > 2 ) {
    throw runtime_error( "unexpected fd count: " + to_string( count ) );
  }

  // XXX rely on "hw" driver behavior where PCM fd is always first

  pollfd pollfds[2];
  alsa_check_easy( snd_pcm_poll_descriptors( pcm_, pollfds, count ) );
  return PCMFD { CheckSystemCall( "dup AudioInterface fd", dup( pollfds[0].fd ) ) };
}

void AudioInterface::initialize()
{
  check_state( SND_PCM_STATE_OPEN );

  snd_pcm_uframes_t buffer_size;

  /* set desired hardware parameters */
  {
    struct params_deleter
    {
      void operator()( snd_pcm_hw_params_t* x ) const { snd_pcm_hw_params_free( x ); }
    };
    unique_ptr<snd_pcm_hw_params_t, params_deleter> params { [] {
      snd_pcm_hw_params_t* x = nullptr;
      snd_pcm_hw_params_malloc( &x );
      return notnull( "snd_pcm_hw_params_malloc", x );
    }() };

    alsa_check_easy( snd_pcm_hw_params_any( pcm_, params.get() ) );

    alsa_check_easy( snd_pcm_hw_params_set_rate_resample( pcm_, params.get(), false ) );
    alsa_check_easy( snd_pcm_hw_params_set_access( pcm_, params.get(), SND_PCM_ACCESS_RW_INTERLEAVED ) );
    alsa_check_easy( snd_pcm_hw_params_set_format( pcm_, params.get(), SND_PCM_FORMAT_FLOAT_LE ) );
    alsa_check_easy( snd_pcm_hw_params_set_channels( pcm_, params.get(), 2 ) );
    alsa_check_easy( snd_pcm_hw_params_set_rate( pcm_, params.get(), 48000, 0 ) );
    alsa_check_easy( snd_pcm_hw_params_set_period_size( pcm_, params.get(), 48, 0 ) );
    alsa_check_easy( snd_pcm_hw_params_set_buffer_size( pcm_, params.get(), 384 ) );

    /* apply hardware parameters */
    alsa_check_easy( snd_pcm_hw_params( pcm_, params.get() ) );

    /* save buffer size */
    alsa_check_easy( snd_pcm_hw_params_get_buffer_size( params.get(), &buffer_size ) );
  }

  check_state( SND_PCM_STATE_PREPARED );

  /* set desired software parameters */
  {
    struct params_deleter
    {
      void operator()( snd_pcm_sw_params_t* x ) const { snd_pcm_sw_params_free( x ); }
    };
    unique_ptr<snd_pcm_sw_params_t, params_deleter> params { [] {
      snd_pcm_sw_params_t* x = nullptr;
      snd_pcm_sw_params_malloc( &x );
      return notnull( "snd_pcm_sw_params_malloc", x );
    }() };

    alsa_check_easy( snd_pcm_sw_params_current( pcm_, params.get() ) );

    alsa_check_easy( snd_pcm_sw_params_set_avail_min( pcm_, params.get(), 48 ) );
    alsa_check_easy( snd_pcm_sw_params_set_period_event( pcm_, params.get(), true ) );
    alsa_check_easy( snd_pcm_sw_params_set_start_threshold( pcm_, params.get(), 1 ) );
    alsa_check_easy( snd_pcm_sw_params_set_stop_threshold( pcm_, params.get(), buffer_size - 1 ) );

    alsa_check_easy( snd_pcm_sw_params( pcm_, params.get() ) );
  }

  check_state( SND_PCM_STATE_PREPARED );
}

bool AudioInterface::update()
{
  snd_pcm_sframes_t avail_tmp, delay_tmp;

  const auto ret = snd_pcm_avail_delay( pcm_, &avail_tmp, &delay_tmp );

  if ( ret < 0 ) {
    avail_ = delay_ = 0;
    cerr << name() << ": " << snd_strerror( ret ) << "\n";
    return true;
  }

  if ( avail_tmp < 0 or delay_tmp < 0 ) {
    throw runtime_error( "avail < 0 or delay < 0" );
  }

  avail_ = avail_tmp;
  delay_ = delay_tmp;

  return false;
}

void AudioInterface::recover()
{
  drop();
  prepare();
}

string AudioInterface::name() const
{
  return annotation_ + "[" + interface_name_ + "]";
}

AudioInterface::~AudioInterface()
{
  try {
    alsa_check( "snd_pcm_close(" + name() + ")", snd_pcm_close( pcm_ ) );
  } catch ( const exception& e ) {
    cerr << "Exception in destructor: " << e.what() << endl;
  }
}

snd_pcm_state_t AudioInterface::state() const
{
  return snd_pcm_state( pcm_ );
}

void AudioInterface::check_state( const snd_pcm_state_t expected_state )
{
  const auto actual_state = state();
  if ( expected_state != actual_state ) {
    snd_pcm_sframes_t avail, delay;
    snd_pcm_avail_delay( pcm_, &avail, &delay );
    throw runtime_error( name() + ": expected state " + snd_pcm_state_name( expected_state ) + " but state is "
                         + snd_pcm_state_name( actual_state ) + " with avail=" + to_string( avail )
                         + " and delay=" + to_string( delay ) );
  }
}

void AudioInterface::start()
{
  check_state( SND_PCM_STATE_PREPARED );
  alsa_check( "snd_pcm_start(" + name() + ")", snd_pcm_start( pcm_ ) );
  check_state( SND_PCM_STATE_RUNNING );
}

void AudioInterface::drop()
{
  alsa_check( "snd_pcm_drop(" + name() + ")", snd_pcm_drop( pcm_ ) );
  check_state( SND_PCM_STATE_SETUP );
}

void AudioInterface::prepare()
{
  check_state( SND_PCM_STATE_SETUP );
  alsa_check( "snd_pcm_prepare(" + name() + ")", snd_pcm_prepare( pcm_ ) );
  check_state( SND_PCM_STATE_PREPARED );
}

void AudioInterface::play( const ChannelPair& audio, const size_t play_until_sample )
{
  if ( play_until_sample <= cursor() ) {
    return;
  }
  const size_t frames_available_to_write = play_until_sample - cursor();

  array<pair<float, float>, 32768> sample_buffer;

  const size_t frames_to_write = min( frames_available_to_write, sample_buffer.size() );
  if ( not frames_to_write ) {
    return;
  }

  for ( unsigned int i = 0; i < frames_to_write; i++ ) {
    sample_buffer.at( i ).first = audio.ch1().safe_get( cursor() + i );
    sample_buffer.at( i ).second = audio.ch2().safe_get( cursor() + i );
  }

  const auto frames_written = snd_pcm_writei( pcm_, sample_buffer.data(), frames_to_write );
  poll_fd_.register_write();
  if ( frames_written == -EPIPE or frames_written == -ESTRPIPE ) {
    drop();
    recover();
    return;
  } else if ( frames_written < 0 ) {
    throw alsa_error( "snd_pcm_writei", frames_written );
  }

  cerr << "wrote " << frames_written << "\n";
  cursor_ += frames_written;
}

short AudioInterface::transform_revents( const short revents_in, const short dir ) const
{
  unsigned short ret;
  pollfd the_pollfd { poll_fd_.fd_num(), dir, revents_in };
  alsa_check_easy( snd_pcm_poll_descriptors_revents( pcm_, &the_pollfd, 1, &ret ) );
  return ret;
}
