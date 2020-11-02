find_package(Doxygen REQUIRED)

set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
doxygen_add_docs(docs ${CMAKE_SOURCE_DIR}/include/hicma)