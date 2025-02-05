CC = nvcc
CFLAGS = -O3 -arch=sm_80 -Xcompiler -Wall -Xcompiler -Wextra
PROFILE_DIR = profiling_results

# Source file (change this to use a different CUDA file)
SRC = tc_red.cu
TARGET = $(basename $(SRC))

# Profiler flags
NVPROF_FLAGS = --metrics achieved_occupancy,gld_efficiency,gst_efficiency,shared_load_throughput
NSYS_FLAGS = --stats=true --trace=cuda,nvtx --force-overwrite true
NCU_FLAGS = --kernel-id ::stencil_3d_gpu --metrics smsp__sass_inst_executed_op_shared_st.vec,dram__bytes.sum

.PHONY: all clean profile profile_nsys profile_ncu

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<
	

clean:
	rm -f $(TARGET) *.nsys-rep *.ncu-rep
	rm -rf $(PROFILE_DIR)

$(PROFILE_DIR):
	mkdir -p $(PROFILE_DIR)

profile: $(TARGET) | $(PROFILE_DIR)
	nvprof --log-file $(PROFILE_DIR)/nvprof.log --metrics gpu_time,achieved_occupancy,shared_utilization --events all ./$(TARGET)

profile_nsys: $(TARGET) | $(PROFILE_DIR)
	nsys profile $(NSYS_FLAGS) -o $(PROFILE_DIR)/profile.nsys-rep ./$(TARGET)

profile_ncu: $(TARGET) | $(PROFILE_DIR)
	ncu $(TARGET) -o $(PROFILE_DIR)/profile.ncu-rep --export $(PROFILE_DIR)/profile.ncu --force-overwrite ./$(TARGET)

profile_set: $(TARGET)
	ncu --set full $(TARGET)

full_profile: profile profile_nsys profile_ncu

help:
	@echo "Available targets:"
	@echo "  all          - Build the executable (default)"
	@echo "  profile      - Run basic nvprof profiling"
	@echo "  profile_nsys - Run Nsight Systems profile"
	@echo "  profile_ncu  - Run Nsight Compute profile"
	@echo "  profile_set  - Run Nsight Compute profile --set full"
	@echo "  full_profile - Run all profiling tools"
	@echo "  clean        - Remove generated files"
