# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

function(mindiesd_prepare_eagle_blaze result)
    set(revision cee7d18649df06467384d275aed40f4c33169188)
    set(MINDIESD_BLAZE_INCLUDE_DIR "$ENV{MINDIESD_BLAZE_INCLUDE_DIR}" CACHE PATH "Offline ops-tensor/include for EagleHadamardQuant")
    if(MINDIESD_BLAZE_INCLUDE_DIR)
        get_filename_component(include_dir "${MINDIESD_BLAZE_INCLUDE_DIR}" ABSOLUTE)
    else()
        include(FetchContent)
        message(STATUS "Fetching ops-tensor ${revision} for EagleHadamardQuant. Set MINDIESD_BLAZE_INCLUDE_DIR for an offline build.")
        FetchContent_Declare(eagle_ops_tensor
            GIT_REPOSITORY https://gitcode.com/cann/ops-tensor.git
            GIT_TAG ${revision}
            GIT_SUBMODULES ""
            SOURCE_SUBDIR include/blaze)
        if(CMAKE_VERSION VERSION_LESS 3.18)
            FetchContent_GetProperties(eagle_ops_tensor)
            if(NOT eagle_ops_tensor_POPULATED)
                FetchContent_Populate(eagle_ops_tensor)
            endif()
        else()
            FetchContent_MakeAvailable(eagle_ops_tensor)
        endif()
        set(include_dir "${eagle_ops_tensor_SOURCE_DIR}/include")
    endif()

    if(NOT IS_DIRECTORY "${include_dir}/blaze" OR NOT EXISTS "${include_dir}/../LICENSE")
        message(FATAL_ERROR "EagleHadamardQuant requires ops-tensor ${revision}: expected ${include_dir}/blaze and ${include_dir}/../LICENSE.")
    endif()
    set(headers
        blaze/epilogue/tile/arch35/mx_quant.h
        blaze/gemm/block/block_scheduler_rotate_quant.h
        blaze/gemm/tile/arch35/fill_ub.h
        blaze/gemm/utils/buffer_manager.h
        blaze/gemm/utils/common_utils.h
        blaze/gemm/utils/layout_utils.h
        blaze/gemm/utils/sync.h)
    set(stage "${CMAKE_CURRENT_BINARY_DIR}/eagle_include")
    set(hashes "")
    foreach(header IN LISTS headers)
        set(source "${include_dir}/${header}")
        if(NOT EXISTS "${source}")
            message(FATAL_ERROR "Missing ${source}. Set MINDIESD_BLAZE_INCLUDE_DIR to the include directory of ops-tensor ${revision}.")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
        file(READ "${source}" contents)
        string(REPLACE "\r\n" "\n" contents "${contents}")
        string(SHA256 hash "${contents}")
        string(APPEND hashes "${hash}")

        # The installed OPP compiles kernels on another host. Resolve transitive
        # Blaze includes relative to each header so they travel with the kernel.
        get_filename_component(parent "${header}" DIRECTORY)
        file(RELATIVE_PATH rel "${include_dir}/${parent}" "${include_dir}/blaze")
        string(REPLACE "#include \"blaze/" "#include \"${rel}/" contents "${contents}")
        file(MAKE_DIRECTORY "${stage}/${parent}")
        file(WRITE "${stage}/${header}" "${contents}")
    endforeach()
    string(SHA256 fingerprint "${hashes}")
    if(NOT fingerprint STREQUAL "e70616b70abb2ac241c24208d67daf2f5076f8ef5f1a97241a162bc4e53dadaf")
        message(FATAL_ERROR "Blaze headers in ${include_dir} differ from ops-tensor ${revision}. Check MINDIESD_BLAZE_INCLUDE_DIR.")
    endif()
    set(dest "${IMPL_INSTALL_DIR}/ascendc/eagle_hadamard_quant/arch35")
    install(DIRECTORY "${stage}/blaze" DESTINATION "${dest}")
    install(FILES "${include_dir}/../LICENSE" DESTINATION "${dest}/blaze")
    set(${result} "${stage}" PARENT_SCOPE)
endfunction()

set(eagle_options --cce-auto-sync=off --cce-no-dcache-flush -DENABLE_CV_COMM_VIA_SSBUF=true)
if("ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
    mindiesd_prepare_eagle_blaze(eagle_blaze_include)
    list(APPEND eagle_options "-I${eagle_blaze_include}")
endif()
# Preserve the ASC tensor API include paths used by the arch35 implementation.
foreach(eagle_include IN ITEMS
    asc/include asc/include/basic_api asc/include/utils include pkg_inc compiler/tikcpp/include
    compiler/ascendc/include compiler/ascendc/include/basic_api/impl
    compiler/ascendc/include/basic_api/interface compiler/ascendc/include/highlevel_api/impl
    compiler/ascendc/include/highlevel_api/tiling compiler/ascendc/impl/aicore/basic_api)
    list(APPEND eagle_options "-I${ASCEND_CANN_PACKAGE_PATH}/${eagle_include}")
endforeach()
if("$ENV{MINDIESD_EAGLE_REUSE_L0B}" STREQUAL "0")
    list(APPEND eagle_options -DMINDIESD_EAGLE_REUSE_L0B=0)
endif()
foreach(feature BATCH_L1 PAIR_FIXPIPE FOUR_COLUMNS)
    if("$ENV{MINDIESD_EAGLE_${feature}}" STREQUAL "0")
        list(APPEND eagle_options "-DMINDIESD_EAGLE_${feature}=0")
    else()
        list(APPEND eagle_options "-DMINDIESD_EAGLE_${feature}=1")
    endif()
endforeach()
add_ops_compile_options(OP_NAME EagleHadamardQuant OPTIONS ${eagle_options})

target_sources(op_host_aclnnInner PRIVATE eagle_hadamard_quant_def.cpp)
target_sources(optiling PRIVATE eagle_hadamard_quant_tiling.cpp)
target_sources(opsproto PRIVATE eagle_hadamard_quant_proto.cpp)
target_include_directories(optiling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(opsproto PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(opapi PRIVATE
    op_api/aclnn_eagle_hadamard_quant.cpp
    op_api/eagle_hadamard_quant.cpp)
target_include_directories(opapi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/op_api)

install(FILES op_api/aclnn_eagle_hadamard_quant.h DESTINATION ${ACLNN_INC_INSTALL_DIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/arch35/
    DESTINATION ${IMPL_INSTALL_DIR}/ascendc/eagle_hadamard_quant/arch35
    FILES_MATCHING PATTERN "*.h")
