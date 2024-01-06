NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\"

ifdef dbg
    NVCC_FLAGS  += -g -G
else
    NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE         = trajectory
MOD         = modules
OBJDIR      = obj
MOD_OBJDIR  = $(OBJDIR)/modules

# Object files with directory prefix
OBJS        = $(OBJDIR)/cuda_traj.o \
              $(OBJDIR)/trajectory.o \
              $(OBJDIR)/harness.o \
              $(MOD_OBJDIR)/atmosphere.o \
              $(MOD_OBJDIR)/NA.o

default: $(EXE)


$(OBJDIR)/trajectory.o: trajectory.cpp trajectory.h
	@mkdir -p $(OBJDIR)
	$(NVCC) -c -o $@ trajectory.cpp $(NVCC_FLAGS)

$(OBJDIR)/cuda_traj.o: cuda_traj.cu cuda_traj.h
	@mkdir -p $(OBJDIR)
	$(NVCC) -c -o $@ cuda_traj.cu $(NVCC_FLAGS)

$(OBJDIR)/harness.o: test_harness.cpp trajectory.h
	@mkdir -p $(OBJDIR)
	$(NVCC) -c -o $@ test_harness.cpp $(NVCC_FLAGS)

$(MOD_OBJDIR)/%.o: $(MOD)/%.cpp
	@mkdir -p $(MOD_OBJDIR)
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

$(EXE): $(OBJS)
	$(NVCC) $(OBJS) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf $(OBJDIR)/*.o $(MOD_OBJDIR)/*.o $(EXE)
