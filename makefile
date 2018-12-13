CC = mpicc
MPIFLAGS = -lm
EXECUTABLES = project

.SUFFIXES: .c.o
	$(CC) $(MPIFLAGS) -c $^ 

project: project.o
	$(CC) $(MPIFLAGS) -o $@ $^ -DDEBUG=1

.PHONY: clean
clean:
	$(RM) *.o $(EXECUTABLES)

