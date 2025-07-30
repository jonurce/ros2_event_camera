#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "event_camera_codecs::codec" for configuration ""
set_property(TARGET event_camera_codecs::codec APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(event_camera_codecs::codec PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libcodec.so"
  IMPORTED_SONAME_NOCONFIG "libcodec.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS event_camera_codecs::codec )
list(APPEND _IMPORT_CHECK_FILES_FOR_event_camera_codecs::codec "${_IMPORT_PREFIX}/lib/libcodec.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
