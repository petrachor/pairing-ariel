cmake_minimum_required(VERSION 3.9)
project(PAIRING)

execute_process(COMMAND cargo clean ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE CARGO_STDOUT ERROR_VARIABLE CARGO_STDERR)
execute_process(COMMAND cargo build --release --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml OUTPUT_VARIABLE CARGO_STDOUT ERROR_VARIABLE CARGO_STDERR)

string(REPLACE "\n" ";" CARGO_RESULTS "${CARGO_STDOUT} ${CARGO_STDERR}")
foreach(CARGO_LINE ${CARGO_RESULTS})
  message("# ${CARGO_LINE}")
  if("${CARGO_LINE}" MATCHES "^note: library: .*")
    string(REPLACE "note: library: " "" CARGO_LINE ${CARGO_LINE})
    list(APPEND RUST_DEPS ${CARGO_LINE})
  endif()
endforeach()

## Find rust library
file(GLOB RUST_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/target/release/*.a)

## Use C fixture?
# file(GLOB RUST_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/build_c/release/*extern_static*)

## Setup fake import
add_library(Pairing STATIC IMPORTED)
set_property(TARGET Pairing PROPERTY IMPORTED_LOCATION ${RUST_LIBS})
set_property(TARGET Pairing PROPERTY ${RUST_LIBS})
#add_dependencies(Pairing PairingLib)
## Finalize...
message("RUST LIBS: ${RUST_LIBS}")
message("RUST DEPS: ${RUST_DEPS}")


install(FILES ${RUST_LIBS} DESTINATION ${CMAKE_LIB_TARGET_LOCATION})
#install(FILES ${RUST_DEPS} DESTINATION ${CMAKE_LIB_TARGET_LOCATION})
#install(FILES ${RUST_DEPS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#target_link_libraries(PAIRING PairingLib)
