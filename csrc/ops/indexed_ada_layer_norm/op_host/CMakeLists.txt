add_ops_compile_options(
        OP_NAME IndexedAdaLayerNorm
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

target_sources(op_host_aclnn PRIVATE
        indexed_ada_layer_norm_def.cpp
)

target_sources(optiling PRIVATE
        indexed_adaln_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        indexed_adaln_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        indexed_adaln_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
)
