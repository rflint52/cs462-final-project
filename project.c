/* Authors: Ryan Flint, Chris Dean
   COSC 462 Final Project

   Synopsis: ... */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>
#define DEBUG 1
#define SIZE 128


void part1(); //Serial matrix-matrix multiplication. This part is finished.
void part2(int rank, int size); //Use parallel algorithm discussed in class (i.e. Lecture 13 & 14 in the OneNote) to do the multiplication
void part3(int rank, int size); //Use Cannon's algorithm (also in Lecture 13 & 14 in the OneNote) to do the multiplication


//Take two arrays of size 'size' and compute the dot product
double dot(double a[], double b[], int size);

int main(int argc, char **argv) {

	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != 4 && size != 16 && size != 64 && size != 256) {
		fprintf(stderr, "Error: Number of processors must be one of 4, 16, 64, 256\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	//if (rank == 0) part1();
	part2(rank, size);
	//part3(rank, size);

	MPI_Finalize();
}

void part1() {

	//Serially do matrix-matrix multiplication
	double matrixA[SIZE][SIZE];
	double matrixB[SIZE][SIZE];
	double matrixBTranspose[SIZE][SIZE];
	double result[SIZE][SIZE];

	double entry;
	double timeElapsed;

	int i, j;

	struct timeval *t1, *t2;

	t1 = (struct timeval *) malloc( sizeof(struct timeval) );
	t2 = (struct timeval *) malloc( sizeof(struct timeval) );

	//Generate the two matrices
	entry = 0.001;
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			matrixA[i][j] = entry;
			matrixB[i][j] = entry * 2.0;
			entry += 0.001;
		}
	}
	if (DEBUG) {
		printf("matrix a\n");
		for (i = 0; i < SIZE; i++) {
			for (j = 0; j < SIZE; j++) {
				printf("\t%lf", matrixA[i][j]);
			}
			printf("\n");
		}
	}


	if (DEBUG) {
		printf("matrix b\n");
		for (i = 0; i < SIZE; i++) {
			for (j = 0; j < SIZE; j++) {
				printf("\t%lf", matrixB[i][j]);
			}
			printf("\n");
		}
	}

	gettimeofday(t1, NULL);

	//Get the transpose of matrix B so we don't needlessly regenerate columns for the dot product
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			matrixBTranspose[i][j] = matrixB[j][i];
		}
	}

	//gettimeofday(t1, NULL);


	//Do the computations
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			result[i][j] = dot(matrixA[i], matrixBTranspose[j], SIZE);
		}
	}

	if (DEBUG) {
		printf("result\n");
		for (i = 0; i < SIZE; i++) {
			for (j = 0; j < SIZE; j++) {
				printf("\t%lf", result[i][j]);
			}
			printf("\n");
		}
	}


	gettimeofday(t2, NULL);

	//Display the timing
	timeElapsed = (t2->tv_sec - t1->tv_sec) * 1000.0;
	timeElapsed += (t2->tv_usec - t1->tv_usec) / 1000.0;
	timeElapsed /= 1000.0;

	printf("Time elapsed for serial computation: %lf\n", timeElapsed);

	free(t1);
	free(t2);
}

void part2(int rank, int size) {
	//Use parallel algorithm discussed in class (i.e. Lecture 13 & 14 in the OneNote) to do the multiplication
	double **initSubBlockA, **initSubBlockB;
	int sbSideLen = SIZE / sqrt(size);

	double entry;
	int i, j;

	if (DEBUG) printf("Sub block side length: %d\n", sbSideLen);

	//Set up initial mappings. Each processor owns a (SIZE / sqrt(size) ) * (SIZE / sqrt(size) ) non-overlapping sub-block of matrices A and
	//B, respectively
	initSubBlockA = (double **) malloc( sizeof(double *) * sbSideLen);
	initSubBlockB = (double **) malloc( sizeof(double *) * sbSideLen);

	entry = 0.001 + (sbSideLen * 128 * rank * 0.001) + ( (0.001 * rank * sbSideLen) % sbSideLen); //Starting point for the current processor

	for (i = 0; i < sbSideLen; i++) {
		initSubBlockA[i] = (double *) malloc( sizeof(double) * sbSideLen);
		initSubBlockB[i] = (double *) malloc( sizeof(double) * sbSideLen);
		entry = 0.001 * i * SIZE * rank + 0.001; //This isn't going to work...
		for (j = 0; j < sbSideLen; j++) {
			initSubBlockA[i][j] = entry;
			initSubBlockB[i][j] = entry * 2.0;
			entry += 0.001;
		}
	}






}
double dot(double a[], double b[], int size) {
	double rv;
	int i;

	rv = 0;

	for (i = 0; i < size; i++) rv += (a[i] * b[i]);

	return rv;
}
