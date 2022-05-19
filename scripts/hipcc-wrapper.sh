#!/bin/sh
## Note: this script is DEPRECATED. please use 

# this wrapper script is used for nvlink/hipcc to ignore link static libs twice.
REPLACE="../src/arch_hip/lib/libmd_arch_hip.a"
REPLACE_T="../../src/arch_hip/lib/libmd_arch_hip.a"

is_first=false
is_first_t=false
LINKER=/opt/compilers/rocm/4.2.0/bin/hipcc
newcmd="$LINKER"

for arg in $@
do
  case $arg in
    $REPLACE) # remove in building misa-md binary
      if [ "$is_first" = false ]; then
        newcmd="$newcmd $arg"
      fi
      is_first=true
      ;;
    $REPLACE_T) # remove in building test binary
      if [ "$is_first_t" = false ]; then
        newcmd="$newcmd $arg"
      fi
      is_first_t=true
      ;;
    *)
      newcmd="$newcmd $arg";;
  esac
done

# Finally execute the new command
exec $newcmd
