set(__WF_SIZE__ ${GPU_WAVEFRONT_SIZE})

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/md_hip_building_config.h.in"
        "${PROJECT_BINARY_DIR}/generated/md_hip_building_config.h"
)

# install the generated file
install(FILES "${PROJECT_BINARY_DIR}/generated/md_hip_building_config.h"
        DESTINATION "include"
        )
