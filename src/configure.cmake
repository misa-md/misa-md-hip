set(__WF_SIZE__ ${GPU_WAVEFRONT_SIZE})

if(MD_USE_NEWTONS_THIRD_LAW_FLAG)
        set(USE_NEWTONS_THIRD_LAW ON)
endif()

# set kernel strategy.
string(TOLOWER ${MD_KERNEL_STRATEGY} MD_KERNEL_STRATEGY_LOWER)
if (MD_KERNEL_STRATEGY_LOWER MATCHES "default")
        set(MD_KERNEL_STRATEGY_DEFAULT ON)
elseif (MD_KERNEL_STRATEGY_LOWER MATCHES "thread_atom")
        set(MD_KERNEL_STRATEGY_THREAD_ATOM ON)
elseif (MD_KERNEL_STRATEGY_LOWER MATCHES "wf_atom")
        set(MD_KERNEL_STRATEGY_WF_ATOM ON)
elseif (MD_KERNEL_STRATEGY_LOWER MATCHES "block_atom")
        set(MD_KERNEL_STRATEGY_BLOCK_ATOM ON)
else ()
        MESSAGE(FATAL_ERROR "unsupported kernel strategy ${MD_KERNEL_STRATEGY}")
endif ()
MESSAGE(STATUS "current kernel strategy is: ${MD_KERNEL_STRATEGY}")


# set optimization options.
string(TOLOWER ${MD_OPTIMIZE_OPTION} MD_OPTIMIZE_OPTION_LOWER)
if (MD_OPTIMIZE_OPTION_LOWER MATCHES "default")
        set(MD_OPTIMIZE_OPTION_DEFAULT ON)
elseif()
        # check is a number
        if (NOT MD_OPTIMIZE_OPTION_LOWER MATCHES "^[0-9]+$" )
                MESSAGE(FATAL_ERROR "`MD_OPTIMIZE_OPTION` must be a number.")
        endif ()
endif ()


configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/md_hip_building_config.h.in"
        "${PROJECT_BINARY_DIR}/generated/md_hip_building_config.h"
)

# install the generated file
install(FILES "${PROJECT_BINARY_DIR}/generated/md_hip_building_config.h"
        DESTINATION "include"
        )
