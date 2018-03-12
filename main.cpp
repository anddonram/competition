/*
 * main.cpp
 *
 *  Created on: 7/3/2018
 *      Author: andres
 */
#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include "competition.h"
#include "test_competition.h"

int main(int argc, char** argv)
{
	bool version2=false;
	bool test=false;
	char c='\0';
	while ((c = getopt (argc, argv, "spr:a:m:tv")) != -1) {
		switch (c) {
		case 's':
			competition::set_random_seed(true);
			break;
		case 'p':
			competition::set_print(true);
			break;
		case 'r':
			competition::set_num_rules(atoi(optarg));
			break;
		case 'a':
			competition::set_alphabet(atoi(optarg));
			break;
		case 'm':
			competition::set_max_obj_per_rule(atoi(optarg));
			break;
		case 't':
			test=true;
			break;
		case 'v':
			version2=true;
			break;
		default:
			break;
		}
	}

	if(test){
//		matrices::test_matrix_pow_2();
//		matrices::test_matrix_product();
//		matrices::test_matrix_pow();
//		matrices::test_competition_matrix();
	//	test_compare_cpu_2();
	//	test_compare_cpu_gpu_2();
	//	test_compare_cpu_gpu();
	//	test_normalize_partition_2();
		test_normalize_partition();


		test_compare_cpu_gpu_random();
		//test_compare_cpu_gpu_2_random();


		return 0;
	}
	if (competition::random_seed){
		srand(time(NULL));
	}else{
		srand(1);
	}

	int *alphabet=new int[competition::ALPHABET];
	int *partition=new int [competition::NUM_RULES];
	competition::reset_partition(partition, alphabet);

	//Generate random rules
	int *rules_size = new int[competition::NUM_RULES+1];
	int total_lhs=competition::initialize_rules (rules_size, competition::NUM_RULES+1);

	int *lhs_object=new int[total_lhs];
	competition::initialize_lhs(lhs_object,total_lhs);
	competition::print_header();

	competition::print_rules(rules_size, lhs_object);

	std::cout<< "--- CPU partition ---"<< std::endl;

	clock_t cpu_startTime, cpu_endTime;

    double cpu_ElapseTime=0;
    cpu_startTime = clock();

    competition::make_partition(partition,rules_size,lhs_object,total_lhs,alphabet);

    cpu_endTime = clock();

    cpu_ElapseTime = ((cpu_endTime - cpu_startTime)/(double)CLOCKS_PER_SEC);

    std::cout<< "CPU time: "<< cpu_ElapseTime <<std::endl;

    competition::print_partition(partition, alphabet);

    competition::reset_partition( partition, alphabet);

	std::cout<< "--- GPU partition ---"<< std::endl;
	competition::make_partition_gpu(partition,rules_size,lhs_object,total_lhs,alphabet,version2);

	competition::print_partition(partition, alphabet);



	delete [] rules_size;
	delete [] lhs_object;
	delete [] partition;
	delete [] alphabet;
	return 0;

}




