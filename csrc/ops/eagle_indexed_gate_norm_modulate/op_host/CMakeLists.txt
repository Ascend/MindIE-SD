add_ops_compile_options(
    OP_NAME EagleIndexedGateNormModulate
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations -Werror -fpermissive
)
set(eagle_indexed_gate_norm_modulate_depends eagle_indexed_gate_norm_modulate PARENT_SCOPE)
target_sources(op_host_aclnn PRIVATE eagle_indexed_gate_norm_modulate_def.cpp)
target_sources(optiling PRIVATE eagle_indexed_gate_norm_modulate_tiling.cpp)
if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE eagle_indexed_gate_norm_modulate_tiling.cpp)
endif ()
target_sources(opsproto PRIVATE eagle_indexed_gate_norm_modulate_proto.cpp)
target_include_directories(optiling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc)
