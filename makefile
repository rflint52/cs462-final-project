CC = mpicc
MPIFLAGS = -lm
EXECUTABLES = project

.SUFFIXES: .c.o
	$(CC) $(MPIFLAGS) -c $^

project: project.o
	$(CC) $(MPIFLAGS) -o $@ $^

.PHONY: clean
clean:
	$(RM) *.o $(EXECUTABLES)

