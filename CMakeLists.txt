cmake_minimum_required ( VERSION 2.8.5 )
cmake_policy ( SET CMP0048 NEW )
project ( Beep )

include ( etc/build_type.cmake )
include ( etc/cflags.cmake )
include ( etc/clang_format.cmake )
include ( etc/tests.cmake )

find_package ( PkgConfig )

pkg_check_modules ( ALSA REQUIRED alsa )
include_directories ( ${ALSA_INCLUDE_DIRS} )
add_compile_options ( ${ALSA_CFLAGS} )

pkg_check_modules ( Sndfile REQUIRED sndfile )
include_directories ( ${Sndfile_INCLUDE_DIRS} )
add_compile_options ( ${Sndfile_CFLAGS} )

include_directories ("${PROJECT_SOURCE_DIR}/src/util")
include_directories ("${PROJECT_SOURCE_DIR}/src/audio")

add_subdirectory ("${PROJECT_SOURCE_DIR}/src/util")
add_subdirectory ("${PROJECT_SOURCE_DIR}/src/audio")
add_subdirectory ("${PROJECT_SOURCE_DIR}/src/frontend")
add_subdirectory ("${PROJECT_SOURCE_DIR}/src/tests")
