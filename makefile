MPICC = mpicc
MPIFLAGS = #Not sure why we'd need flags
EXECUTABLES = project

.SUFFIXES: .c.o
	$(MPICC) $(MPIFLAGS) -c $^

project: project.o
	$(MPICC) $(MPIFLAGS) -o $@ $^

.PHONY: clean
clean:
	$(RM) *.o $(EXECUTABLES)

