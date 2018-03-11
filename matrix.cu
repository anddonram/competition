/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "competition.h"

namespace matrices{


/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

/*
 * Set the matrix to the identity
 * */
__global__ void matrix_identity(unsigned long int * result,unsigned int matrix_size){
	 unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		 if(index<matrix_size*matrix_size){
			 unsigned int row=index/matrix_size;
			 unsigned int column=index%matrix_size;
			 result[index]=row==column;
		 }
}

/*
 * Multiplies two squared matrices
 * */
__global__ void matrix_product(unsigned long int* a,unsigned long int* b,unsigned long int * result,unsigned int matrix_size){

	 unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	 if(index<matrix_size*matrix_size){

		 unsigned int row=index/matrix_size;
		 unsigned int column=index%matrix_size;
		 unsigned long int res=0;
		 for(int i=0;i<matrix_size;i++){
			 res+= a[i+row*matrix_size]*b[column+i*matrix_size];
		 }
		 __syncthreads();
		 //printf("%i index writes %lu \n",index,res);
		 result[index]=res;
	 }

}
__global__ void matrix_competition(unsigned long int* result,unsigned int matrix_size,int* rules_size, int*lhs_object,int NUM_RULES){
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx < NUM_RULES*NUM_RULES){
		unsigned rule_a=idx/NUM_RULES;
		unsigned rule_b=idx%NUM_RULES;


		unsigned rule_id_begin=rules_size[rule_a];
		unsigned rule_id_end=rules_size[rule_a+1];


		unsigned rule_begin=rules_size[rule_b];
		unsigned rule_end=rules_size[rule_b+1];

		for (unsigned int j=rule_begin; j<rule_end; j++){
			unsigned object_to_compare=lhs_object[j];
			for (unsigned int k=rule_id_begin; k<rule_id_end; k++){
				if(lhs_object[k]!=object_to_compare){
				//printf("%u and %u prints to %u \n",object_to_compare,lhs_object[k],object_to_compare*matrix_size+lhs_object[k]);
				result[object_to_compare*matrix_size+lhs_object[k]]=1;
				}
			}
		}
	}

}

int test_competition_matrix(){

	//Generate random rules
	int *rules_size = new int[competition::NUM_RULES+1];
	int total_lhs=competition::initialize_rules (rules_size, competition::NUM_RULES+1);

	int *lhs_object=new int[total_lhs];
	competition::initialize_lhs(lhs_object,total_lhs);

	std::cout<< "------ Testing Competency Matrix ------"<< std::endl;

	competition::print_header();
	competition::print_rules(rules_size,lhs_object);

	unsigned const int MATRIX_SIZE=competition::ALPHABET;
	unsigned const int MATRIX_ELEMENTS=MATRIX_SIZE*MATRIX_SIZE;
	unsigned long int *matrix=new unsigned long int[MATRIX_ELEMENTS];
	unsigned long int *result=new unsigned long int[MATRIX_ELEMENTS];

	unsigned long int* d_matrix;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_matrix, sizeof(long int)*MATRIX_ELEMENTS));
	CUDA_CHECK_RETURN(cudaMemset(d_matrix,0,sizeof(long int)*MATRIX_ELEMENTS));
	int * d_rules_size;
	int * d_lhs_object;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_rules_size,sizeof(int)*(competition::NUM_RULES+1)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_lhs_object, sizeof(int)*total_lhs));

	CUDA_CHECK_RETURN(cudaMemcpy(d_rules_size, rules_size, sizeof(int)*(competition::NUM_RULES+1), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_lhs_object, lhs_object,sizeof(int)*total_lhs, cudaMemcpyHostToDevice));


	static const int BLOCK_SIZE = 256;
	const int blockCount = (competition::NUM_RULES*competition::NUM_RULES+BLOCK_SIZE-1)/BLOCK_SIZE;

	matrix_competition<<<blockCount,BLOCK_SIZE>>>(d_matrix,MATRIX_SIZE,d_rules_size,d_lhs_object,competition::NUM_RULES);


	cudaMemcpy(matrix, d_matrix, sizeof(long int)*MATRIX_ELEMENTS, cudaMemcpyDeviceToHost);

	matrix_pow(matrix,result,MATRIX_SIZE,MATRIX_SIZE);

	print_matrix(matrix,MATRIX_SIZE,MATRIX_SIZE);
	print_matrix(result,MATRIX_SIZE,MATRIX_SIZE);


	CUDA_CHECK_RETURN(cudaFree(d_rules_size));
	CUDA_CHECK_RETURN(cudaFree(d_lhs_object));
	CUDA_CHECK_RETURN(cudaFree(d_matrix));


	delete [] matrix;
	delete [] result;
	delete [] lhs_object;
	delete [] rules_size;

	return 0;
}
/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int test_matrix_product(){
	unsigned const int MATRIX_SIZE=3;
	unsigned const int MATRIX_ELEMENTS=MATRIX_SIZE*MATRIX_SIZE;
	unsigned long int *matrix=new unsigned long int[MATRIX_ELEMENTS];
	unsigned long int *result=new unsigned long int[MATRIX_ELEMENTS];
	unsigned long int* d_matrix;
	unsigned long int* d_result;
	for(int i=0;i<MATRIX_ELEMENTS;i++){
		matrix[i]=i;
	}

	cudaMalloc((void **)&d_matrix, sizeof(long int)*MATRIX_ELEMENTS);
	cudaMalloc((void **)&d_result, sizeof(long int)*MATRIX_ELEMENTS);
	cudaMemcpy(d_matrix, matrix, sizeof(long int)*MATRIX_ELEMENTS, cudaMemcpyHostToDevice);

	static const int BLOCK_SIZE = 256;
	const int blockCount = (MATRIX_ELEMENTS+BLOCK_SIZE-1)/BLOCK_SIZE;


	matrix_product<<<blockCount,BLOCK_SIZE>>>(d_matrix,d_matrix,d_result,MATRIX_SIZE);
	cudaMemcpy(result, d_result, sizeof(long int)*MATRIX_ELEMENTS, cudaMemcpyDeviceToHost);

	print_matrix(matrix,MATRIX_SIZE,MATRIX_SIZE);
	print_matrix(result,MATRIX_SIZE,MATRIX_SIZE);

	cudaFree(d_matrix);
	cudaFree(d_result);
	delete [] matrix;
	delete [] result;
	return 0;
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 *
 */
int test_matrix_pow(){
	unsigned const int MATRIX_SIZE=20;
	unsigned const int POW=3;
	unsigned const int MATRIX_ELEMENTS=MATRIX_SIZE*MATRIX_SIZE;
	unsigned long int *matrix=new unsigned long int[MATRIX_ELEMENTS];
	unsigned long int *result=new unsigned long int[MATRIX_ELEMENTS];

	for(int i=0;i<MATRIX_ELEMENTS;i++){
		matrix[i]=i;
	}

	matrix_pow(matrix,result,MATRIX_SIZE,POW);

	print_matrix(matrix,MATRIX_SIZE,MATRIX_SIZE);
	print_matrix(result,MATRIX_SIZE,MATRIX_SIZE);

	delete [] matrix;
	delete [] result;
	return 0;
}

void matrix_pow_rec(unsigned int pow, unsigned int matrix_size,
		unsigned long int* d_aux, unsigned long int* d_result, unsigned long int* d_a) {
	static const int BLOCK_SIZE = 256;
	const int blockCount = (matrix_size*matrix_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int val = pow;

	matrix_identity<<<blockCount, BLOCK_SIZE>>>(d_aux,matrix_size);
	cudaMemcpy(d_result, d_a, sizeof(long int)*matrix_size*matrix_size, cudaMemcpyDeviceToDevice);

//	printf("aux in ptr %u\n",d_aux);
//	printf("result in ptr %u\n",d_result);
	while (val > 1){
	  if (val%2!=0){
		  matrix_product<<<blockCount, BLOCK_SIZE>>>(d_aux, d_result, d_aux,
				matrix_size);

		val--;
	  }
		matrix_product<<<blockCount, BLOCK_SIZE>>>(d_result, d_result, d_result,
				matrix_size);
		val /= 2;
	}

	matrix_product<<<blockCount, BLOCK_SIZE>>>(d_aux, d_result, d_result,
			matrix_size);



//	unsigned long int* d_aux_ptr=d_result;
//	unsigned long int* d_aux_res=d_aux;
//	unsigned long int* temp_ptr=NULL;
//	while (val > 1){
//		printf("val %u\n",val);
//		temp_ptr=d_aux_ptr;
// 		d_aux_ptr=d_aux_res;
//		d_aux_res=temp_ptr;
//		printf("aux ptr %u\n",d_aux_ptr);
//		printf("res ptr %u\n",d_aux_res);
//
//		matrix_product<<<blockCount, BLOCK_SIZE>>>(d_aux_ptr, d_aux_ptr, d_aux_res,
//				matrix_size);
//
//		if (val % 2 != 0) {
//			temp_ptr=d_aux_ptr;
//	 		d_aux_ptr=d_aux_res;
//			d_aux_res=temp_ptr;
//			matrix_product<<<blockCount, BLOCK_SIZE>>>(d_aux_ptr, d_a, d_aux_res,
//								matrix_size);
//			val--;
//
//
//		}
//
//		val/=2;
//
//		cudaDeviceSynchronize();
//
//		}
//
//	printf("finally, aux ptr %u\n",d_aux_ptr);
//	printf("finally, res ptr %u\n",d_aux_res);
//	if(d_aux_ptr==d_result){
//		cudaMemcpy(d_result,d_aux, sizeof(long int)*matrix_size*matrix_size,cudaMemcpyDeviceToDevice);
//	}
}

void matrix_pow(const unsigned long int *a,unsigned long int *result,unsigned int matrix_size,unsigned int pow){
	unsigned int MATRIX_ELEMENTS=matrix_size*matrix_size;

	unsigned long int* d_a;
	unsigned long int* d_aux;
	unsigned long int* d_result;

	cudaMalloc((void **)&d_a, sizeof(long int)*MATRIX_ELEMENTS);
	cudaMalloc((void **)&d_aux, sizeof(long int)*MATRIX_ELEMENTS);
	cudaMalloc((void **)&d_result, sizeof(long int)*MATRIX_ELEMENTS);

	cudaMemcpy(d_a, a, sizeof(long int)*MATRIX_ELEMENTS, cudaMemcpyHostToDevice);

	matrix_pow_rec(pow, matrix_size, d_aux, d_result, d_a);

	cudaMemcpy(result, d_result, sizeof(long int)*MATRIX_ELEMENTS, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_aux);
	cudaFree(d_result);


}
//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const unsigned long int *A, int nr_rows_A, int nr_cols_A) {

      for(int i = 0; i < nr_rows_A; ++i){
          for(int j = 0; j < nr_cols_A; ++j){
              std::cout << A[i * nr_rows_A + j] << " ";
          }
          std::cout << std::endl;
      }
     std::cout << std::endl;
 }


int test_matrix_pow_2(){


	std::cout<< "------ Testing Matrix Pow ------"<< std::endl;


	unsigned const int MATRIX_SIZE=4;
	unsigned const int MATRIX_ELEMENTS=MATRIX_SIZE*MATRIX_SIZE;
	unsigned long int values[MATRIX_ELEMENTS]={
			0,0,1,0,
			0,0,0,1,
			1,0,0,1,
			0,1,1,0};
	unsigned long int *matrix=values;

	unsigned long int *result=new unsigned long int[MATRIX_ELEMENTS];


	for(int i=1;i<6;i++){
		matrix_pow(matrix,result,MATRIX_SIZE,i);

		print_matrix(result,MATRIX_SIZE,MATRIX_SIZE);
	}

	//delete [] matrix;
	delete [] result;

	return 0;
}

}
