/* Authors: Ryan Flint, Chris Dean
   COSC 462 Final Project

Synopsis: ... */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#define DEBUG 0
#define DEBUG_RANK -1
#define SIZE 256
#define CSV 1


//Return 'double' because returning time required for each
double part1(); //Serial matrix-matrix multiplication. This part is finished.
double part2(int rank, int size); //Use parallel algorithm discussed in class (i.e. Lecture 13 & 14 in the OneNote) to do the multiplication
double part3(int rank, int size); //Use Cannon's algorithm (also in Lecture 13 & 14 in the OneNote) to do the multiplication


//Take two arrays of size 'size' and compute the dot product
double dot(double a[], double b[], int size);
//Take two blocks and multiply them and add their result to c
double mult_blocks(double * a, double * b, double * c, int block_size);

int main(int argc, char **argv) {

	int rank, size;
	double p1, p2, p3;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != 4 && size != 16 && size != 64 && size != 256) {
		fprintf(stderr, "Error: Number of processors must be one of 4, 16, 64, 256\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if(CSV && rank == 0) printf("%d,", size);

	if (rank == 0) p1 = part1();
	MPI_Barrier(MPI_COMM_WORLD);
	p2 = part2(rank, size);
	MPI_Barrier(MPI_COMM_WORLD);
	p3 = part3(rank, size);

	if(CSV && rank == 0) printf("\n", size);

	MPI_Finalize();
}

double part1() {

	//Serially do matrix-matrix multiplication
	double matrixA[SIZE][SIZE];
	double matrixB[SIZE][SIZE];
	double matrixBTranspose[SIZE][SIZE];
	double result[SIZE][SIZE];

	double entry;
	double timeElapsed;
	double t1, t2;

	int i, j;


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

	t1 = MPI_Wtime();
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


	t2 = MPI_Wtime();

	//Display the timing
	/*
	   timeElapsed = (t2->tv_sec - t1->tv_sec) * 1000.0;
	   timeElapsed += (t2->tv_usec - t1->tv_usec) / 1000.0;
	   timeElapsed /= 1000.0; */

	timeElapsed = t2 - t1;

	if(!CSV) printf("Time elapsed for serial computation: %lf\n", timeElapsed);
	else printf("%lf,", timeElapsed);

	return timeElapsed;

}

double part2(int rank, int size) {
	//Use parallel algorithm discussed in class (i.e. Lecture 13 & 14 in the OneNote) to do the multiplication
	double **blockMatrixA, **blockMatrixB;
	double **recvSubBlockA, **recvSubBlockB, **localResultC;
	double **subBTranspose;
	double ***finalResult; //Array of matrices--You think an array of vectors is nasty, Chris?

	double t1, t2;
	double timeElapsed;

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


	//Array of matrices... This is disgusting...
	finalResult = (double ***) malloc( sizeof(double ***) * size);
	for (i = 0; i < size; i++) {
		finalResult[i] = (double **) malloc( sizeof(double *) * sbSideLen);
		for (j = 0; j < sbSideLen; j++) {
			finalResult[i][j] = (double *) malloc( sizeof(double) * sbSideLen);
		}
	}



	t1 = MPI_Wtime();

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
			if (DEBUG && DEBUG_RANK == rank ) {
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

	if (leftSource % origMatSLen == 0) leftSource = (rank / origMatSLen) * origMatSLen;
	if (leftDest % origMatSLen == (origMatSLen - 1) || leftDest < 0 ) leftDest = (rank / origMatSLen) * origMatSLen + origMatSLen - 1;

	if (upSource >= size) upSource = rank % origMatSLen;
	if (upDest < 0) upDest = (origMatSLen * origMatSLen) - origMatSLen + (rank % origMatSLen);

#if 0
	if (DEBUG  ) {
		printf("origMatSLen: %d\n", origMatSLen);
		printf("Processor %d's leftSource = %d\n", rank, leftSource);
		printf("Processor %d's leftDest = %d\n", rank, leftDest);
		printf("Processor %d's upsource = %d\n", rank, upSource);
		printf("Processor %d's updest = %d\n", rank, upDest);
		//while (1);
		MPI_Barrier(MPI_COMM_WORLD);
	}
#endif

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

	if (DEBUG && DEBUG_RANK == rank) {
		printf("Processor %d's final subblock result C\n", rank);
		for (i = 0; i < sbSideLen; i++) {
			for (j = 0; j < sbSideLen; j++) {
				printf("\t%lf (%d)", localResultC[i][j], rank);
			}
			printf("\n");
		}
	}

	//MPI_Barrier(MPI_COMM_WORLD);

	//Get everything back on rank 0 at the end
	if (rank != 0) {
		for (i = 0; i < sbSideLen; i++) {
			MPI_Send(localResultC[i], sbSideLen, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
	} else {

		//Get rank 0's block
		for (i = 0; i < sbSideLen; i++) {
			for (j = 0; j < sbSideLen; j++) {
				finalResult[0][i][j] = localResultC[i][j];
			}
		}

		//Get the blocks from the other ranks
		for (i = 1; i < size; i++) {
			for (j = 0; j < sbSideLen; j++) {
				MPI_Recv(finalResult[i][j], sbSideLen, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}

		t2 = MPI_Wtime();

		//Print out the final result

		if (DEBUG) {

			printf("[Processor 0] Printing the final result matrix...\n\n");

			for (i = 0; i < origMatSLen; i++) {
				for (j = 0; j < sbSideLen; j++) {
					for (k = 0; k < origMatSLen; k++) {
						for (l = 0; l < sbSideLen; l++) {
							printf("\t%lf", finalResult[i * origMatSLen + k][j][l]);
						}
					}
					printf("\n");
				}
			}

		}

		if(!CSV) printf("Time elapsed for the simple parallel matrix-matrix multiplication: %lf\n", t2 - t1);
		else printf("%lf,", t2 - t1);

		//Free stuff
		for (i = 0; i < size; i++) {
			for (j = 0; j < sbSideLen; j++) {
				free(finalResult[i][j]);
			}
			free(finalResult[i]);
		}
		free(finalResult);

		return (t2 - t1);

	}

	//Free other stuff
	for (i = 0; i < sbSideLen; i++) {
		free(recvSubBlockA[i]);
		free(recvSubBlockB[i]);
		free(subBTranspose[i]);
		free(localResultC[i]);
	}

	free(recvSubBlockA);
	free(recvSubBlockB);
	free(subBTranspose);
	free(localResultC);

	return -1;

}

void print_mat(double ** mat, int size);
double part3(int rank, int size)
{
	int dims[2];
	int periodic[2];
	int coords[2];
	int i;
	int k;
	int j;
	int block_size;
	int num_blocks;
	int block_length;
	int blocks_per_side;
	int row;
	int col;
	int row_;
	int col_;
	int left, right, up, down;
	double entry;
	double timeElapsed;
	double t1;
	double t2;
	double ** blocks;
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
	blocks_per_side = sqrt(size);

	a = malloc(sizeof(double) * block_size);
	b = malloc(sizeof(double) * block_size);
	c = calloc(sizeof(double),  block_size);

	dims[0] = blocks_per_side;
	dims[1] = blocks_per_side;
	periodic[0] = 1;
	periodic[1] = 1;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 1, &cart_comm);
	//get our rank and cvoordinates
	MPI_Comm_rank(cart_comm, &rank);

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
		t1 = MPI_Wtime();
		//assemble our thing int blocks and send them out
		for(i = 1; i < size; i++)
		{
			row_ = i / blocks_per_side;
			col_ = i % blocks_per_side;
			for(j = 0; j < block_length; j++)
			{
				for(k = 0; k < block_length; k++)
				{
					a[j * block_length + k] = mat_a[row_ * block_length + j][col_ * block_length + k];
					b[j * block_length + k] = mat_b[row_ * block_length + j][col_ * block_length + k];
				}
			}
			if(i != 0)
			{
				MPI_Send(a, block_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
				MPI_Send(b, block_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			}
		}
		for(j = 0; j < block_length; j++)
		{
			for(k = 0; k < block_length; k++)
			{
				a[j * block_length + k] = mat_a[j][k];
				b[j * block_length + k] = mat_b[j][k];
			}
		}
	}
	else
	{
		//recieve all our blocks
		MPI_Recv(a, block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
		MPI_Recv(b, block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
	}

	char * buff = malloc(10000);
	char * b2 = malloc(100);

	//get our coordinates
	MPI_Cart_coords(cart_comm, rank, 2, coords);

	//do initial alignment and multiplication
	MPI_Cart_shift(cart_comm, 1, coords[0], &left, &right);
	MPI_Cart_shift(cart_comm, 0, coords[1], &up, &down);
	MPI_Sendrecv_replace(a, block_size, MPI_DOUBLE, left, 11, right, 11, cart_comm, MPI_STATUS_IGNORE);
	MPI_Sendrecv_replace(b, block_size, MPI_DOUBLE, up, 11, down, 11, cart_comm, MPI_STATUS_IGNORE);

	mult_blocks(a, b, c, block_length);

	//do shifting and multiplying
	for(i = 1; i < blocks_per_side; i++)
	{
		MPI_Cart_shift(cart_comm, 1, 1, &left,&right);
		MPI_Cart_shift(cart_comm, 0, 1, &up,&down);
		MPI_Sendrecv_replace(a, block_size, MPI_DOUBLE, left, 11, right, 11, cart_comm, MPI_STATUS_IGNORE);
		MPI_Sendrecv_replace(b, block_size, MPI_DOUBLE, up, 11, down, 11, cart_comm, MPI_STATUS_IGNORE);
		mult_blocks(a, b, c, block_length);
	}

	//done now gather

	if(rank == 0)
	{
		blocks = malloc(size * sizeof(double*));
		blocks[0] = c;
		for(i = 0; i < size; i++)
		{
			if(i)
			{
				//recieve the blocks
				blocks[i] = malloc(sizeof(double*) * block_size);
				MPI_Recv(blocks[i], block_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &stat);
			}

			//assemble into matrix
			for(j = 0; j < block_length; j++)
			{
				for(k = 0; k < block_length; k++)
				{
					row_ = (i / blocks_per_side) * block_length;
					col_ = (i % blocks_per_side) * block_length;
					mat_a[row_ + j][col_ + k] = blocks[i][j * block_length + k];
				}
			}


		}

		if(DEBUG)
		{
			printf("C:\n");
			print_mat(mat_a, SIZE);
		}


		t2 = MPI_Wtime();

		timeElapsed = t2 - t1;

		if(!CSV) printf("Time elapsed for cannon's algorithm: %lf\n", timeElapsed);
		else printf("%lf", timeElapsed);

	}
	else
	{
		MPI_Send(c, block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}


	return t2 - t1;
}

void print_mat(double ** mat, int size)
{
	int i;
	int j;
	for(i = 0; i < size; i++)
	{
		for(j = 0; j < size; j++)
		{
			printf("\t%.6f ", mat[i][j]);
		}
		printf("\n");
	}
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
