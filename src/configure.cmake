set(__WF_SIZE__ ${GPU_WAVEFRONT_SIZE})

# set kernel strategy.
string(TOLOWER ${MD_KERNEL_STRATEGY} MD_KERNEL_STRATEGY_LOWER)
if (MD_KERNEL_STRATEGY_LOWER MATCHES "default")
        set(MD_KERNEL_STRATEGY_DEFAULT ON)
elseif (MD_KERNEL_STRATEGY_LOWER MATCHES "thread_atom")
        set(MD_KERNEL_STRATEGY_THREAD_ATOM ON)
elseif (MD_KERNEL_STRATEGY_LOWER MATCHES "wf_atom")
        set(MD_KERNEL_STRATEGY_WF_ATOM ON)
else ()
        MESSAGE(FATAL_ERROR "unsupported kernel strategy ${MD_KERNEL_STRATEGY}")
endif ()
MESSAGE(STATUS "current kernel strategy is: ${MD_KERNEL_STRATEGY}")
 

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/md_hip_building_config.h.in"
        "${PROJECT_BINARY_DIR}/generated/md_hip_building_config.h"
)

# install the generated file
install(FILES "${PROJECT_BINARY_DIR}/generated/md_hip_building_config.h"
        DESTINATION "include"
        )
