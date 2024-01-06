FORTRAN_COMPILER = gfortran

MOD = modules
OBJ = obj


# main program

main: 
# link the programs
	$(FORTRAN_COMPILER) -c  run_traj.f90  -I $(MOD) -o $(OBJ)/$@.o
# compile the program with opt lvl 3
	$(FORTRAN_COMPILER) -O3 $(OBJ)/*.o -o main.out
mods:
	$(FORTRAN_COMPILER) -c $(MOD)/COESA.f90 -J $(MOD) -o $(OBJ)/COESA.o
	$(FORTRAN_COMPILER) -c $(MOD)/RK4.f90 -J $(MOD) -o $(OBJ)/RK4.o
	$(FORTRAN_COMPILER) -c $(MOD)/NA.f90 -J $(MOD) -o $(OBJ)/NA.o
clean:
	# rm main.out
	rm *.o
	

# %.o:%.f90
#     $(FORTRAN_COMPILER) $(FFLAGS) -c $^ -o $@d