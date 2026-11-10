# The source QBMM uses CANN 9 APIs. Keep its C++17 requirement local to these
# object libraries; the shared SD host targets keep their existing standard.
add_ops_compile_options(OP_NAME EagleQuantBatchMatmulSwiglu
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -fpermissive)

target_sources(op_host_aclnn PRIVATE eagle_quant_batch_matmul_swiglu_def.cpp)

# Keep the generated ND API; NZ needs the descriptor-preserving L0 launcher.
target_sources(opapi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_api/aclnn_eagle_quant_batch_matmul_swiglu_weight_nz.cpp)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../op_api/aclnn_eagle_quant_batch_matmul_swiglu_weight_nz.h
    DESTINATION packages/vendors/${VENDOR_NAME}/op_api/include)

add_library(eagle_qbmm_tiling OBJECT
    op_tiling/eagle_quant_batch_matmul_swiglu_tiling.cpp
    op_tiling/eagle_quant_batch_matmul_swiglu_tiling_base.cpp
    op_tiling/eagle_quant_batch_matmul_swiglu_basic_tiling.cpp
    op_tiling/eagle_quant_batch_matmul_swiglu_tiling_cache.cpp
    op_tiling/platform_util.cpp
    deps/common/src/op_host/op_cache_tiling.cpp
    deps/common/src/legacy_common_manager.cpp
    deps/matmul/common/op_host/op_tiling/hash.cpp
    deps/matmul/common/op_host/op_tiling/debug_tiling.cpp)
add_library(eagle_qbmm_proto OBJECT
    eagle_quant_batch_matmul_swiglu_infershape.cpp
    deps/matmul/common/op_host/matmul_common_infershape.cpp)

foreach(qbmm_target eagle_qbmm_tiling eagle_qbmm_proto)
    set_target_properties(${qbmm_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(${qbmm_target} PRIVATE -std=gnu++17 -fvisibility=hidden
        -include ${CMAKE_CURRENT_SOURCE_DIR}/qbmm_compat.h)
    target_compile_definitions(${qbmm_target} PRIVATE LOG_CPP PROCESS_LOG NN_ENABLE_DLOPEN_LEGACY)
    target_link_libraries(${qbmm_target} PRIVATE
        $<BUILD_INTERFACE:intf_pub> $<BUILD_INTERFACE:ops_utils_tiling_headers>)
    target_include_directories(${qbmm_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/include/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/tiling
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/base
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/base)
endforeach()
target_sources(optiling PRIVATE $<TARGET_OBJECTS:eagle_qbmm_tiling>)
target_link_libraries(optiling PRIVATE ${CMAKE_DL_LIBS})
target_sources(opsproto PRIVATE $<TARGET_OBJECTS:eagle_qbmm_proto>)
if(NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE $<TARGET_OBJECTS:eagle_qbmm_tiling>)
endif()
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/deps/
    DESTINATION ${IMPL_INSTALL_DIR}/ascendc/eagle_quant_batch_matmul_swiglu/deps
    FILES_MATCHING PATTERN "*.h")
