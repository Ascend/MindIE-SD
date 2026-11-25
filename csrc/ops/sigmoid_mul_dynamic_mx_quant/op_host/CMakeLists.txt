set(sigmoid_mul_dynamic_mx_quant_depends sigmoid_mul_dynamic_mx_quant PARENT_SCOPE)
target_sources(op_host_aclnn PRIVATE sigmoid_mul_dynamic_mx_quant_def.cpp)
target_sources(optiling PRIVATE sigmoid_mul_dynamic_mx_quant_tiling.cpp)
target_sources(opsproto PRIVATE sigmoid_mul_dynamic_mx_quant_proto.cpp)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE sigmoid_mul_dynamic_mx_quant_tiling.cpp)
endif ()
