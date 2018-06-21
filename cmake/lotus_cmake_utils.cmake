function(add_whole_archive_flag lib output_var)
  if(MSVC)
    # In MSVC, we will add whole archive in default.
    set(${output_var} -WHOLEARCHIVE:$<SHELL_PATH:$<TARGET_FILE:${lib}>> PARENT_SCOPE)
  else()
    # Assume everything else is like gcc
    set(${output_var} "-Wl,--whole-archive $<TARGET_FILE:${lib}> -Wl,--no-whole-archive" PARENT_SCOPE)
  endif()
endfunction()
