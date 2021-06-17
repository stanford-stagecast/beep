add_library (stats STATIC "stats_printer.cc")

add_executable (beep-example "beep-example.cc")
target_link_libraries (beep-example stats)
target_link_libraries (beep-example audio)
target_link_libraries (beep-example util)

target_link_libraries ("beep-example" ${ALSA_LDFLAGS})
target_link_libraries ("beep-example" ${ALSA_LDFLAGS_OTHER})

target_link_libraries ("beep-example" ${Sndfile_LDFLAGS})
target_link_libraries ("beep-example" ${Sndfile_LDFLAGS_OTHER})
