/*
 * competition.h
 *
 *  Created on: 7/3/2018
 *      Author: andres
 */

#ifndef COMPETITION_H_
#define COMPETITION_H_
namespace competition{


extern bool print;
extern bool random_seed;
extern int MAX_OBJECTS_PER_RULE;
extern int ALPHABET;
extern int NUM_RULES;


void print_header();
void print_partition(int* partition, int* alphabet);
void print_comparing_partition(int* partition, int* alphabet,int* partition2, int* alphabet2);
void print_rules(int* rules_size, int* lhs_object);
void reset_partition(int* partition,int* alphabet);
int initialize_rules(int *data, int size);
void initialize_lhs(int *data, int size);
void make_partition(int* partition, int* rules_size, int*lhs_object, int total_lhs,int * alphabet);
void make_partition_2(int* partition, int* rules_size, int*lhs_object, int total_lhs,int * alphabet);
void make_partition_gpu(int* partition, int* rules_size, int*lhs_object, int total_lhs,int * alphabet,bool version2=false);

bool check_compete(int block_a,int block_b, int* rules_size,int * lhs_object);


void set_print(bool set);
void set_random_seed(bool set);
void set_num_rules(int set);
void set_max_obj_per_rule(int set);
void set_alphabet(int set);

void compare_partition(int* partition, int* alphabet,int* partition2, int* alphabet2);
int* normalize_partition(int* partition);
}
namespace matrices{
void print_matrix(const unsigned long int *A, int nr_rows_A, int nr_cols_A);
int test_matrix_product();
int test_matrix_pow();
int test_competition_matrix();
int test_matrix_pow_2();
void matrix_pow(const long unsigned int *a,unsigned int long *result,unsigned int matrix_size,unsigned int pow);
}

#endif /* COMPETITION_H_ */
