/* Authors: Ryan Flint, Chris Dean
   COSC 462 Final Project

Synopsis: ... */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>
#define DEBUG 1
#define DEBUG_RANK 1
#define SIZE 4


void part1(); //Serial matrix-matrix multiplication. This part is finished.
void part2(int rank, int size); //Use parallel algorithm discussed in class (i.e. Lecture 13 & 14 in the OneNote) to do the multiplication
void part3(int rank, int size); //Use Cannon's algorithm (also in Lecture 13 & 14 in the OneNote) to do the multiplication


//Take two arrays of size 'size' and compute the dot product
double dot(double a[], double b[], int size);
//Take two blocks and multiply them and add their result to c
double mult_blocks(double * a, double * b, double * c, int block_size);

int main(int argc, char **argv) {

	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != 4 && size != 16 && size != 64 && size != 256) {
		fprintf(stderr, "Error: Number of processors must be one of 4, 16, 64, 256\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (rank == 0) part1();
	MPI_Barrier(MPI_COMM_WORLD);
	part2(rank, size);
	MPI_Barrier(MPI_COMM_WORLD);
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
	double **blockMatrixA, **blockMatrixB;
	double **recvSubBlockA, **recvSubBlockB, **localResultC;
	double **subBTranspose;

	//Remember to free all this stuff when done

	int sbSideLen = SIZE / sqrt(size);
	int origMatSLen = sqrt(size);

	double startingPoint, entry;
	int i, j, k, l;

	int leftSource, leftDest;
	int upSource, upDest;

	recvSubBlockA = (double **) malloc( sizeof(double *) * sbSideLen);
	recvSubBlockB = (double **) malloc( sizeof(double *) * sbSideLen);
	localResultC = (double **) malloc( sizeof(double *) * sbSideLen);
	subBTranspose = (double **) malloc( sizeof(double *) * sbSideLen);
	for (i = 0; i < sbSideLen; i++) {
		recvSubBlockA[i] = (double *) malloc( sizeof(double) * sbSideLen);
		recvSubBlockB[i] = (double *) malloc( sizeof(double) * sbSideLen);
		localResultC[i] = (double *) malloc( sizeof(double) * sbSideLen);
		subBTranspose[i] = (double *) malloc( sizeof(double) * sbSideLen);
	}


	//Generate the entire matrices block by block on rank 0, then split up work as neccessary
	if (rank == 0) {
		blockMatrixA = (double **) malloc(sizeof(double *) * sbSideLen);
		blockMatrixB = (double **) malloc(sizeof(double *) * sbSideLen);

		for (i = 0; i < sbSideLen; i++) {
			for (j = 0; j < sbSideLen; j++) {
				blockMatrixA[i] = (double *) malloc( sizeof(double *) * sbSideLen);
				blockMatrixB[i] = (double *) malloc( sizeof(double *) * sbSideLen);
			}
		}


		for (i = 0; i < origMatSLen; i++) {
			startingPoint = ( i * sbSideLen * SIZE * 0.001) + 0.001;
			for (j = 0; j < origMatSLen; j++) {
				entry = startingPoint + (j * sbSideLen * 0.001);
				for (k = 0; k < sbSideLen; k++) {

					for (l = 0; l < sbSideLen; l++) {
						blockMatrixA[k][l] = entry;
						blockMatrixB[k][l] = entry * 2.0;

						//Set up rank 0's sub-block
						if ( (i * origMatSLen + j) == 0 ) {
							recvSubBlockA[k][l] = entry;
							recvSubBlockB[k][l] = entry * 2.0;
						}

						entry += 0.001;
					}
					entry += (SIZE * 0.001) - (sbSideLen * 0.001);

					//Send this row of the sublock to the proper processor (i * origMatSLen + j)
					if ( (i * origMatSLen + j) != 0 ) {
						//if (DEBUG) printf("Rank 0 doing a send\n");
						MPI_Send(blockMatrixA[k], sbSideLen, MPI_DOUBLE, (i * origMatSLen + j), 0, MPI_COMM_WORLD);
						MPI_Send(blockMatrixB[k], sbSideLen, MPI_DOUBLE, (i * origMatSLen + j), 0, MPI_COMM_WORLD);
					}
				}
				if (DEBUG && 0) {
					printf("Printing matrix A subblock %d\n", i * origMatSLen + j);
					for (k = 0; k < sbSideLen; k++) {
						for (l = 0; l < sbSideLen; l++) {
							printf("\t%lf", blockMatrixA[k][l]);
						}
						printf("\n");
					}
				}
			}
		}

		for (i = 0; i < sbSideLen; i++) {
			free(blockMatrixA[i]);
			free(blockMatrixB[i]);
		}
		free(blockMatrixA);
		free(blockMatrixB);
	} else { //If rank != 0, get the subblocks from rank 0
		for (i = 0; i < sbSideLen; i++) {
			MPI_Recv(recvSubBlockA[i], sbSideLen, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (DEBUG && DEBUG_RANK == rank) {
				printf("Rank %d got a subblock row\n", rank);
				for (j = 0; j < sbSideLen; j++) printf("\t%lf", recvSubBlockA[i][j]);
				printf("\n");
			}

			MPI_Recv(recvSubBlockB[i], sbSideLen, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}

	//Now we're ready for repeated local matrix-matrix multiplication. This repeats sqrt(size) (i.e. origMatSLen) times

	for (i = 0; i < sbSideLen; i++) {
		for (j = 0; j < sbSideLen; j++) {
			localResultC[i][j] = 0;
		}
	}

	//Set up source and destination processors
	leftSource = rank + 1;
	leftDest = rank - 1;

	upSource = rank + origMatSLen;
	upDest = rank - origMatSLen;

	if (leftSource == size) leftSource = 0;
	if (leftDest == -1) leftDest = size - 1;

	if (upSource >= size) upSource = rank % origMatSLen;
	if (upDest < 0) upDest = (origMatSLen * origMatSLen) - origMatSLen + (rank % origMatSLen);

	if (DEBUG) {
		printf("origMatSLen: %d\n", origMatSLen);
		printf("Processor %d's upsource = %d\n", rank, upSource);
		printf("Processor %d's updest = %d\n", rank, upDest);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	for (i = 0; i < origMatSLen; i++) {

		//First, compute the local matrix-matrix product and add it to the product that already exists

		if (DEBUG && DEBUG_RANK == rank) {

			printf("[BEGIN] Beginning iteration %d...\n", i);

			printf("Printing rank %d's received subblock A\n", rank);
			for (j = 0; j < sbSideLen; j++) {
				for (k = 0; k < sbSideLen; k++) {
					printf("\t%lf", recvSubBlockA[j][k]);
				}
				printf("\n");
			}

			printf("Printing rank %d's received subblock B\n", rank);
			for (j = 0; j < sbSideLen; j++) {
				for (k = 0; k < sbSideLen; k++) {
					printf("\t%lf", recvSubBlockB[j][k]);
				}
				printf("\n");
			}
		}


		//Get transpose of subblock b to avoid recomputing columns of subblock b
		for (j = 0; j < sbSideLen; j++) {
			for (k = 0; k < sbSideLen; k++) {
				subBTranspose[j][k] = recvSubBlockB[k][j];
			}
		}

		//Do the multiplication
		for (j = 0; j < sbSideLen; j++) {
			for (k = 0; k < sbSideLen; k++) {
				localResultC[j][k] += dot(recvSubBlockA[j], subBTranspose[k], sbSideLen);
			}
		}

		//Now comes the gross part. Matrix A sublocks are transferred to processor p - 1 (wrapping around if needed?), and matrix b subblocks are transferred to processor
		//p - origMatSLen (also wrapping when needed?)

		for (j = 0; j < sbSideLen; j++) {
			MPI_Send(recvSubBlockA[j], sbSideLen, MPI_DOUBLE, leftDest, 0, MPI_COMM_WORLD);
			MPI_Send(recvSubBlockB[j], sbSideLen, MPI_DOUBLE, upDest, 0, MPI_COMM_WORLD);

			for (k = 0; k < sbSideLen; k++) {
				recvSubBlockA[j][k] = 0;
				recvSubBlockB[j][k] = 0;
			}

			MPI_Recv(recvSubBlockA[j], sbSideLen, MPI_DOUBLE, leftSource, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(recvSubBlockB[j], sbSideLen, MPI_DOUBLE, upSource, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}

	//Take care of the last computation
	for (j = 0; j < sbSideLen; j++) {
		for (k = 0; k < sbSideLen; k++) {
			subBTranspose[j][k] = recvSubBlockB[k][j];
		}
	}

	//Do the multiplication
	for (j = 0; j < sbSideLen; j++) {
		for (k = 0; k < sbSideLen; k++) {
			localResultC[j][k] += dot(recvSubBlockA[j], subBTranspose[k], sbSideLen);
		}
	}
	if (DEBUG && rank == DEBUG_RANK) {
		printf("Processor %d's final subblock result C\n", rank);
		for (i = 0; i < sbSideLen; i++) {
			for (j = 0; j < sbSideLen; j++) {
				printf("\t%lf", localResultC[i][j]);
			}
			printf("\n");
		}
	}

	//Get everything back on rank 0 at the end

}

void part3(int rank, int size)
{
	int dims[2];
	int periodic[2];
	int coords[2];
	int i;
	int k;
	int j;
	int x;
	int y;
	int block_size;
	int num_blocks;
	int block_length;
	int partitioned_length;
	int row;
	int col;
	int left, right, up, down;
	double entry;
	double ** mat_a;
	double ** mat_b;
	double * a;
	double * b;
	double * c;
	MPI_Status stat;
	MPI_Comm cart_comm;
	MPI_Comm col_comm;


	block_size = SIZE * SIZE / size;
	block_length = sqrt(block_size);
	partitioned_length = sqrt(size);

	a = malloc(sizeof(double) * block_size);
	b = malloc(sizeof(double) * block_size);
	c = malloc(sizeof(double) * block_size);

	if(rank == 0)
	{
		//were generating the matrix block by block here
		//Generate the two matrices
		mat_a = malloc(sizeof(double *) * SIZE);
		mat_b = malloc(sizeof(double *) * SIZE);
		entry = 0.001;
		for (i = 0; i < SIZE; i++)
		{
			mat_a[i] = malloc(sizeof(double) * SIZE);
			mat_b[i] = malloc(sizeof(double) * SIZE);
			for (j = 0; j < SIZE; j++) {
				mat_a[i][j] = entry;
				mat_b[i][j] = entry * 2.0;
				entry += 0.001;
			}
		}
		for(i = 1; i < size; i++)
		{
			x = i / partitioned_length;
			y = i % partitioned_length;
			for(j = 0; j < partitioned_length; j++)
			{
				for(k = 0; k < sqrt(size); k++)
				{
					a[j * partitioned_length + k] = mat_a[x * block_length + j][y*block_length + k];
					b[j * partitioned_length + k] = mat_b[x * block_length + j][y*block_length + k];
				}
			}
			if(i != 0)
			{
				MPI_Send(a, block_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
				MPI_Send(b, block_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			}
		}
		for(j = 0; j < sqrt(size); j++)
		{
			for(k = 0; k < sqrt(size); k++)
			{
				a[j * partitioned_length + k] = mat_a[j][k];
				b[j * partitioned_length + k] = mat_b[j][k];
			}
		}
	}
	else
	{
		MPI_Recv(a, block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
		MPI_Recv(b, block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
	}

	dims[0] = partitioned_length;
	dims[1] = partitioned_length;
	periodic[0] = 1;
	periodic[1] = 1;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 1, &cart_comm);
	//get our rank and cvoordinates
	MPI_Comm_rank(cart_comm, &rank);
	MPI_Cart_coords(cart_comm, rank, 2, coords);

	MPI_Cart_shift(cart_comm, 1, coords[0], &left, &right);
	MPI_Cart_shift(cart_comm, 0, coords[1], &up, &down);
	MPI_Sendrecv_replace(a, block_size, MPI_DOUBLE, left, 11, right, 11, cart_comm, MPI_STATUS_IGNORE);
	MPI_Sendrecv_replace(b, block_size, MPI_DOUBLE, up, 11, down, 11, cart_comm, MPI_STATUS_IGNORE);

	mult_blocks(a, b, c, block_length);

	for(i=1;i<partitioned_length;i++)
	{
		MPI_Cart_shift(cart_comm, 1, 1, &left,&right);
		MPI_Cart_shift(cart_comm, 0, 1, &up,&down);
		MPI_Sendrecv_replace(a, block_size, MPI_DOUBLE, left, 11, right, 11, cart_comm, MPI_STATUS_IGNORE);
		MPI_Sendrecv_replace(b, block_size, MPI_DOUBLE, up, 11, down, 11, cart_comm, MPI_STATUS_IGNORE);
		mult_blocks(a, b, c, block_length);
	}

	//done now gather


}

double mult_blocks(double * a, double * b, double * c, int block_length)
{
	int i;
	int j;
	int k;

	for(i = 0; i < block_length; i++)
	{
		for(j = 0; j < block_length; j++)
		{
			for(k = 0; k < block_length; k++)
			{
				c[i * block_length + j] += a[i * block_length + k] * b[k * block_length + j];
			}
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
