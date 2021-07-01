#pragma once

#include <alsa/asoundlib.h>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "audio_buffer.hh"
#include "beep/util/file_descriptor.hh"

class PCMFD : public FileDescriptor
{
public:
  using FileDescriptor::FileDescriptor;
  using FileDescriptor::register_read;
  using FileDescriptor::register_write;
};

class AudioInterface
{
  std::string interface_name_, annotation_;
  snd_pcm_t* pcm_;
  PCMFD poll_fd_;

  void check_state( const snd_pcm_state_t expected_state );

  size_t avail_ {}, delay_ {};

  size_t cursor_ {};

  PCMFD get_poll_fd();

public:
  AudioInterface( const std::string_view interface_name,
                  const std::string_view annoatation,
                  const snd_pcm_stream_t stream );

  void initialize();
  void start();
  void prepare();
  void drop();
  void recover();
  bool update();

  snd_pcm_state_t state() const;
  size_t avail() const { return avail_; }
  size_t delay() const { return delay_; }
  const PCMFD& poll_fd() { return poll_fd_; }

  std::string name() const;

  size_t cursor() const { return cursor_; }

  void play( const ChannelPair& audio, const size_t play_until_sample );

  ~AudioInterface();

  short transform_revents( const short revents_in, const short dir ) const;

  /* can't copy or assign */
  AudioInterface( const AudioInterface& other ) = delete;
  AudioInterface& operator=( const AudioInterface& other ) = delete;
};

inline float float_to_dbfs( const float sample_f )
{
  if ( sample_f <= 0.00001 ) {
    return -100;
  }

  return 20 * log10( sample_f );
}

inline float dbfs_to_float( const float dbfs )
{
  if ( dbfs <= -99 ) {
    return 0.0;
  }

  return exp10( dbfs / 20 );
}
