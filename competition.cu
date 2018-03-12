/*
 ============================================================================
 Name        : competition.cu
 Author      : AndresDR
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include "competition.h"
#include <stdio.h>


namespace competition{
__global__ void make_partition_phase_1_kernel_3(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES,int ALPHABET);
__global__ void make_partition_phase_1_5_kernel_3(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES,int ALPHABET);
__global__ void make_partition_phase_2_kernel_3(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES,int ALPHABET);
__global__ void get_partition_kernel(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES);


__global__ void make_partition_phase1_kernel(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES,int ALPHABET,int i);
__global__ void make_partition_phase2_kernel(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES,int ALPHABET,int i);
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

bool print=false;
bool random_seed=false;
int MAX_OBJECTS_PER_RULE = 4;
int ALPHABET=128;
int NUM_RULES = 4096;


void set_print(bool set){
	print=set;
}
void set_random_seed(bool set){
	random_seed=set;
}
void set_num_rules(int set){
	NUM_RULES=set;
}
void set_max_obj_per_rule(int set){
	MAX_OBJECTS_PER_RULE=set;
}
void set_alphabet(int set){
	ALPHABET=set;
}

__device__ bool change=true;
/**
 * https://pdfs.semanticscholar.org/4569/2750cb1bf50da6451e70ae06b6519992e4ec.pdf
 * CUDA kernel that computes partition for overlapping rules
 */
__global__ void make_partition_kernel_3(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES,int ALPHABET) {
	static const int BLOCK_SIZE = 256;
	const int blockCount1 = (NUM_RULES+BLOCK_SIZE-1)/BLOCK_SIZE;
	const int blockCount2 = (ALPHABET+BLOCK_SIZE-1)/BLOCK_SIZE;
	change=true;
	bool exit_loop=false;


	while(change || !exit_loop){

		//If nothing changed, we must give a last pass
		exit_loop=!change;
		change=false;
		make_partition_phase_1_kernel_3<<<blockCount1,BLOCK_SIZE>>>( partition, rules_size, lhs_object, alphabet, NUM_RULES, ALPHABET);
		make_partition_phase_1_5_kernel_3<<<blockCount1,BLOCK_SIZE>>>( partition, rules_size, lhs_object, alphabet, NUM_RULES, ALPHABET);
		make_partition_phase_2_kernel_3<<<blockCount2,BLOCK_SIZE>>>( partition,  rules_size, lhs_object, alphabet, NUM_RULES, ALPHABET);

		cudaDeviceSynchronize();
//		printf("iter:\n");
//		for(int i=0;i<ALPHABET;i++)
//		{
//			printf("%3u, ",i);
//
//		}
//		printf("\n");
//		for(int i=0;i<ALPHABET;i++)
//		{
//			printf("%3u, ",alphabet[i]);
//		}
//		printf("\n");
	}

	get_partition_kernel<<<blockCount1, BLOCK_SIZE>>> (partition,rules_size,lhs_object,alphabet,NUM_RULES);

}
__global__ void make_partition_phase_1_kernel_3(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES,int ALPHABET) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx < NUM_RULES){

		unsigned rule_id_begin=rules_size[idx];
		unsigned rule_id_end=rules_size[idx+1];
		int val=ALPHABET;

		for (unsigned int k=rule_id_begin; k<rule_id_end; k++){
			val=min(val,alphabet[lhs_object[k]]);
		}

		partition[idx]=val;

	}
}
__global__ void make_partition_phase_1_5_kernel_3(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES,int ALPHABET) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx < NUM_RULES){

		unsigned rule_id_begin=rules_size[idx];
		unsigned rule_id_end=rules_size[idx+1];

		unsigned val=partition[idx];

		for (unsigned int j=rule_id_begin; j<rule_id_end; j++){
			int old_val=atomicMin((alphabet+lhs_object[j]),val);
			if(old_val!=val)
				change=true;
		}


	}
}
__global__ void make_partition_phase_2_kernel_3(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES,int ALPHABET) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < ALPHABET){
		int i =alphabet[idx];
		int i_1 =alphabet[i];
		bool must_go_on=false;
		while(i_1!=i)
		{
			must_go_on=true;
			i=i_1;
			i_1=alphabet[i];
		}
		atomicMin(alphabet+idx,i_1);
		if(must_go_on){
			change=true;
		}
	}
}
/**
 * CUDA kernel that computes partition for overlapping rules
 * Each thread takes a rule, then all iterate over each other rule
 * If rule idx matches with rule i (have a common object), then rule idx belongs to the min partition of each
 *
 * It does not work correctly
 *
 */
__global__ void make_partition_kernel_2(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES,int ALPHABET) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx < NUM_RULES){
		unsigned rule_id_begin=rules_size[idx];
		unsigned rule_id_end=rules_size[idx+1];

		for(int i=0;i<NUM_RULES;i++){

			if(i==idx)
				continue;
			unsigned rule_begin=rules_size[i];
			unsigned rule_end=rules_size[i+1];

			for (unsigned int j=rule_begin; j<rule_end; j++){
				unsigned object_to_compare=lhs_object[j];

				for (unsigned int k=rule_id_begin; k<rule_id_end; k++){
					if(object_to_compare==lhs_object[k] ){
						atomicMin(partition+i,partition[idx]);
						atomicMin(partition+idx,partition[i]);
						break;

					}

				}

			}


		}

	}
}
/**
 * CUDA kernel that partition of rules
 * We want to determine the components of a disconnected graph (in the sense of rules having no objects in common)
 * Each thread takes a rule(edge), then we iterate each object(vertex) in the alphabet and
 * set its value in the connected graph to the smallest value among the objects in its rule
 * Finally, each rule takes one object and get its value (component of the graph)
 *
 * It does not work properly
 */

__global__ void make_partition_kernel(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES,int ALPHABET) {
	static const int BLOCK_SIZE = 256;
	const int blockCount1 = (NUM_RULES+BLOCK_SIZE-1)/BLOCK_SIZE;
	const int blockCount2 = (ALPHABET+BLOCK_SIZE-1)/BLOCK_SIZE;
    	for(int i=0;i<ALPHABET;i++){

			make_partition_phase1_kernel<<<blockCount1, BLOCK_SIZE>>> (partition,rules_size,lhs_object,alphabet,NUM_RULES,ALPHABET,i);
			make_partition_phase2_kernel<<<blockCount1, BLOCK_SIZE>>> (partition,rules_size,lhs_object,alphabet,NUM_RULES,ALPHABET,i);
		//	make_partition_phase_2_kernel_3<<<blockCount2,BLOCK_SIZE>>>( partition,  rules_size, lhs_object, alphabet, NUM_RULES, ALPHABET);
			cudaDeviceSynchronize();
		}
		get_partition_kernel<<<blockCount1, BLOCK_SIZE>>> (partition,rules_size,lhs_object,alphabet,NUM_RULES);

}
/*
* First phase: find rules with object i and set the i to the minimum
*/
__global__ void make_partition_phase1_kernel(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES,int ALPHABET,int i) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < NUM_RULES){
		unsigned rule_id_begin=rules_size[idx];
		unsigned rule_id_end=rules_size[idx+1];

//			if(idx==0){
//				printf("iteration %i:\n",i);
//			}
			bool found=false;
			int min_val=ALPHABET;
			for (unsigned int k=rule_id_begin; k<rule_id_end; k++){
				found=found||i==lhs_object[k];
				min_val=min	(alphabet[lhs_object[k]],min_val);
			}

			if(found){
				//printf("matches with rule %i \n",idx);
				//Pass the min for each rule to the object
				atomicMin(alphabet+i,min_val);
			}

	}
}
/**
 * Second phase: find rules with object i and set the other objects to i value
 */
__global__ void make_partition_phase2_kernel(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES,int ALPHABET,int i) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < NUM_RULES){
		unsigned rule_id_begin=rules_size[idx];
		unsigned rule_id_end=rules_size[idx+1];

			bool found=false;
			for (unsigned int k=rule_id_begin; k<rule_id_end; k++){
				found=found||i==lhs_object[k];
			}

			if(found){
				int min_val=alphabet[i];
				//Propagate that min value to all other adjacent objects
				for (unsigned int l=rule_id_begin; l<rule_id_end; l++){
					alphabet[lhs_object[l]]=min_val;
					//printf("set object %i with val %i \n",lhs_object[l],min_val);
				}

			}

	}
}
__global__ void get_partition_kernel(int* partition, int* rules_size, int*lhs_object,int * alphabet,int NUM_RULES) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < NUM_RULES){

		partition[idx]=alphabet[lhs_object[rules_size[idx]]];
		//printf("set rule %i with object and partition %i %i \n",idx,lhs_object[rules_size[idx]],partition[idx]);

	}
}


/**
 * Host function that copies the data and launches the work on GPU
 */
void make_partition_gpu(int* partition, int* rules_size, int*lhs_object, int total_lhs,int * alphabet,bool version2){
	int * d_partition;
	int * d_rules_size;
	int * d_lhs_object;
	int * d_alphabet;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_partition, sizeof(int)*NUM_RULES));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_rules_size,sizeof(int)*(NUM_RULES+1)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_lhs_object, sizeof(int)*total_lhs));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_alphabet, sizeof(int)*ALPHABET));

	CUDA_CHECK_RETURN(cudaMemcpy(d_partition, partition, sizeof(int)*NUM_RULES, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_rules_size, rules_size, sizeof(int)*(NUM_RULES+1), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_lhs_object, lhs_object,sizeof(int)*total_lhs, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_alphabet, alphabet, sizeof(int)*ALPHABET, cudaMemcpyHostToDevice));

	static const int BLOCK_SIZE = 256;
	const int blockCount = (NUM_RULES+BLOCK_SIZE-1)/BLOCK_SIZE;

	clock_t cpu_startTime, cpu_endTime;

    double cpu_ElapseTime=0;
    cpu_startTime = clock();

    if(version2){
    	make_partition_kernel_2<<<blockCount, BLOCK_SIZE>>> (d_partition,d_rules_size,d_lhs_object,d_alphabet,NUM_RULES,ALPHABET);
    }else{
    	make_partition_kernel_3<<<1,1>>>(d_partition,d_rules_size,d_lhs_object,d_alphabet,NUM_RULES,ALPHABET);
    	//make_partition_kernel<<<1,1>>>(d_partition,d_rules_size,d_lhs_object,d_alphabet,NUM_RULES,ALPHABET);
    }
    cudaDeviceSynchronize();

    cpu_endTime = clock();

    cpu_ElapseTime = ((cpu_endTime - cpu_startTime)/(double)CLOCKS_PER_SEC);

    std::cout<< "GPU time: "<< cpu_ElapseTime <<std::endl;


	CUDA_CHECK_RETURN(cudaMemcpy(partition, d_partition, sizeof(int)*NUM_RULES, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(alphabet, d_alphabet, sizeof(int)*ALPHABET, cudaMemcpyDeviceToHost));



	CUDA_CHECK_RETURN(cudaFree(d_partition));
	CUDA_CHECK_RETURN(cudaFree(d_rules_size));
	CUDA_CHECK_RETURN(cudaFree(d_lhs_object));
	CUDA_CHECK_RETURN(cudaFree(d_alphabet));

}

int initialize_rules(int *data, int size)
{
	data[0]=0;
	for (int i = 1; i < size; ++i){

		data[i] =data[i-1]+(rand()%MAX_OBJECTS_PER_RULE)+1;
	}
	return data[size-1];
}
void initialize_lhs(int *data, int size)
{
	for (int i = 0; i < size; ++i){
		data[i] =rand()%ALPHABET;
	}

}
/*
 * Sequential of partition kernel version 1
 * Does not work correctly
 * */
void make_partition_2(int* partition, int* rules_size, int*lhs_object, int total_lhs,int * alphabet){
	for(int i=0;i<ALPHABET;i++){

		int abs_min=alphabet[i];
		for(int j=0;j<NUM_RULES;j++){
			unsigned rule_id_begin=rules_size[j];
			unsigned rule_id_end=rules_size[j+1];

			bool found=false;
			int min_val=ALPHABET;
			for (unsigned int k=rule_id_begin; k<rule_id_end; k++){
				found=found||i==lhs_object[k];
				min_val=min	(alphabet[lhs_object[k]],min_val);
			}
			if(found){
				abs_min=min(min_val,abs_min);
			}
		}
		for(int j=0;j<NUM_RULES;j++){
			unsigned rule_id_begin=rules_size[j];
			unsigned rule_id_end=rules_size[j+1];

			for (unsigned int k=rule_id_begin; k<rule_id_end; k++){

				if (i==lhs_object[k]){
					for (unsigned int l=rule_id_begin; l<rule_id_end; l++){
						alphabet[lhs_object[l]]=abs_min;
					}
					break;
				}

			}
		}
	}
	for(int i=0;i<NUM_RULES;i++){
			partition[i]=alphabet[lhs_object[rules_size[i]]];
	}

}
/**
 * Modified sequential version of partition kernel version 3
 * It works
 */
void make_partition(int* partition, int* rules_size, int*lhs_object, int total_lhs,int * alphabet){
	bool change=true;
	bool exit_loop=false;

	while(change || !exit_loop){

		//If nothing changed, we must give a last pass
		exit_loop=!change;
		change=false;

		for(int i=0;i<NUM_RULES;i++){
			for(int j=i+1;j<NUM_RULES;j++){
				if(partition[j]!=partition[i] && check_compete(i,j,rules_size,lhs_object)){
					if(partition[j]<partition[i]){
						partition[i]=partition[j];
						change=true;
					}else{
						partition[j]=partition[i];
					}
				}
			}
		}
		for(int idx=0;idx<NUM_RULES;idx++){
			int i =partition[idx];
			int i_1 =partition[i];
			bool must_go_on=false;
			while(i_1!=i)
			{
				must_go_on=true;
				i=i_1;
				i_1=partition[i];
			}
			partition[idx]=min(partition[idx],i_1);
			if(must_go_on){
				change=true;
			}
		}
		}
}


bool check_compete(int block_a,int block_b, int* rules_size,int * lhs_object){
	bool res=false;
	for (unsigned int j=rules_size[block_a]; j<rules_size[block_a+1]; j++){
		for (unsigned int k=rules_size[block_b]; k<rules_size[block_b+1]; k++){

			if(lhs_object[j]==lhs_object[k]){
				res=true;
				break;
			}
		}
	}
	return res;
}

void reset_partition(int* partition,int* alphabet) {
	for (int i = 0; i < ALPHABET; i++) {
		alphabet[i] = i;
	}
	//At most, they will be all independent
	for (int i = 0; i < NUM_RULES; i++) {
		partition[i] = i;
	}
}

void print_header(){
	std::cout<< "--- " << NUM_RULES <<" rules generated with at most "
			<< MAX_OBJECTS_PER_RULE<< " objects each and "
			<<ALPHABET <<" objects in alphabet" <<" ---"<< std::endl;

}
void print_rules(int* rules_size, int* lhs_object) {
	if(!print)
		return;
	for (int i = 0; i < competition::NUM_RULES; i++) {
		std::cout << "Rule " << i << std::endl;
		for (int j = rules_size[i]; j < rules_size[i + 1]; j++) {
			std::cout << "\t Object " << lhs_object[j] << std::endl;
		}
	}
}

void print_partition( int* partition, int* alphabet) {
	if(!print)
		return;

	for (int i = 0; i < NUM_RULES; i++) {
		std::cout << "\t Rule " << i << " belongs to part " << partition[i]
				<< std::endl;
	}
//	for (int i = 0; i < ALPHABET; i++) {
//		std::cout << "\t Object " << i << " belongs to part " << alphabet[i]
//				<< std::endl;
//	}
}
void print_comparing_partition(int* partition, int* alphabet,int* partition2, int* alphabet2) {

	for (int i = 0; i < NUM_RULES; i++) {
		std::cout << "\t Rule " << i << " belongs to part: " << partition[i]
		          << "\t | \t " << partition2[i]
				<< std::endl;
	}
//	for (int i = 0; i < ALPHABET; i++) {
//		std::cout << "\t Object " << i << " belongs to part " << alphabet[i]
//				<< "\t | \t " << alphabet2[i]
//				<< std::endl;
//	}
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

void compare_partition(int* partition, int* alphabet,int* partition2, int* alphabet2){

//	for(int i=0;i<ALPHABET;i++){
//		if(alphabet[i]!=alphabet2[i]){
//			std::cout<<"Alphabet not matching"<<std::endl;
//			break;
//		}
//	}

	for(int i=0;i<NUM_RULES;i++){
		if(partition[i]!=partition2[i]){
			std::cout<<"Partition not matching"<<std::endl;
			return;
		}
	}

	std::cout<<"Works great!!!"<<std::endl;
}
int* normalize_partition(int* partition){
	int * trans_partition=new int[NUM_RULES];
	for(int i=0;i<NUM_RULES;i++){
		trans_partition[i]=-1;
	}

	int part_index=0;
	for (int i=0;i<NUM_RULES;i++){
		for(int j=0;j<i;j++){
			if(partition[j]==partition[i]){
				trans_partition[i]=trans_partition[j];
				break;
			}
		}
		if(trans_partition[i]==-1){
			trans_partition[i]=part_index;
			part_index++;
		}
	}

	return trans_partition;
}
}
