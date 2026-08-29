add_ops_compile_options(
        OP_NAME MulAdd
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

set(mul_add_depends mul_add PARENT_SCOPE)
target_sources(op_host_aclnn PRIVATE
        mul_add_def.cpp
)

target_sources(optiling PRIVATE
        mul_add_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        mul_add_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        mul_add_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
)
