option(MD_USE_NEWTONS_THIRD_LAW_FLAG "Enable newton's third law" OFF)

set(GPU_WAVEFRONT_SIZE "64" CACHE STRING "threads number in a wavefront (usually 32 for NVIDIA GPU or 64 for AMD GPU and DCU)")

set(MD_KERNEL_STRATEGY "thread_atom" CACHE STRING "kernel strategy")
set(MD_OPTIMIZE_OPTION "Default" CACHE STRING "optimization option")
