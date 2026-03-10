add_ops_compile_options(
        OP_NAME LaPreprocess
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

set(la_preprocess_depends la_preprocess PARENT_SCOPE)
target_sources(op_host_aclnn PRIVATE
        la_preprocess_def.cpp
)

target_sources(optiling PRIVATE
        la_preprocess_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        la_preprocess_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        la_preprocess_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)