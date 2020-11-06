find_package(Doxygen REQUIRED)

list(APPEND DOXYGEN_ALIASES
  [[OMM="\ref md_pages_open_multi_methods \"OMM\""]]
  [[OMMs="\ref md_pages_open_multi_methods \"OMMs\""]]
  [[ext_hicma="\ref md_pages_extending_hicma \"Extending HiCMA\""]]
)
set(DOXYGEN_VERBATIM_VARS DOXYGEN_ALIASES)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_SOURCE_DIR}/README.md)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
doxygen_add_docs(docs
    ${CMAKE_SOURCE_DIR}/include/hicma
    ${CMAKE_CURRENT_SOURCE_DIR}/pages
)