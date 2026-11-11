add_ops_compile_options(
        OP_NAME AdaRelu2BlockEstimate
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

set(ada_relu2_block_estimate_depends ada_relu2_block_estimate PARENT_SCOPE)

target_sources(op_host_aclnn PRIVATE
        ada_relu2_block_estimate_def.cpp
)

target_sources(optiling PRIVATE
        ada_relu2_block_estimate_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            ada_relu2_block_estimate_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        ada_relu2_block_estimate_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
