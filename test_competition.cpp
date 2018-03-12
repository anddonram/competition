/*
 * test_competition.cpp
 *
 *  Created on: 9/3/2018
 *      Author: andres
 */

#include <stdlib.h>
#include <iostream>
#include "competition.h"
#include "test_competition.h"

int test_compare_cpu_2(){

	int *alphabet=new int[competition::ALPHABET];
	int *partition=new int [competition::NUM_RULES];


	int *alphabet2=new int[competition::ALPHABET];
	int *partition2=new int [competition::NUM_RULES];
	competition::reset_partition(partition,alphabet);
	competition::reset_partition(partition2,alphabet2);

	//Generate random rules
	int *rules_size = new int[competition::NUM_RULES+1];
	int total_lhs=competition::initialize_rules (rules_size, competition::NUM_RULES+1);

	int *lhs_object=new int[total_lhs];
	competition::initialize_lhs(lhs_object,total_lhs);

	std::cout<< "------ Testing CPU competition version 2 ------"<< std::endl;

	competition::print_header();
	competition::print_rules(rules_size,lhs_object);

    competition::make_partition(partition,rules_size,lhs_object,total_lhs,alphabet);
	competition::make_partition_2(partition2,rules_size,lhs_object,total_lhs,alphabet2);

	competition::print_comparing_partition(partition,alphabet,partition2,alphabet2);
	competition::compare_partition(partition,alphabet,partition2,alphabet2);
	std::cout<< std::endl;

	delete [] rules_size;
	delete [] lhs_object;
	delete [] partition;
	delete [] alphabet;
	delete [] partition2;
	delete [] alphabet2;
	return 0;

}


int test_compare_cpu_gpu(){

	int *alphabet=new int[competition::ALPHABET];
	int *partition=new int [competition::NUM_RULES];


	int *alphabet2=new int[competition::ALPHABET];
	int *partition2=new int [competition::NUM_RULES];
	competition::reset_partition(partition,alphabet);
	competition::reset_partition(partition2,alphabet2);

	//Generate random rules
	int *rules_size = new int[competition::NUM_RULES+1];
	int total_lhs=competition::initialize_rules (rules_size, competition::NUM_RULES+1);

	int *lhs_object=new int[total_lhs];
	competition::initialize_lhs(lhs_object,total_lhs);

	std::cout<< "------ Testing GPU competition version 1 ------"<< std::endl;

	competition::print_header();
	competition::print_rules(rules_size,lhs_object);

    competition::make_partition(partition,rules_size,lhs_object,total_lhs,alphabet);
	competition::make_partition_gpu(partition2,rules_size,lhs_object,total_lhs,alphabet2);

	competition::print_comparing_partition(partition,alphabet,partition2,alphabet2);
	competition::compare_partition(partition,alphabet,partition2,alphabet2);
	std::cout<< std::endl;

	delete [] rules_size;
	delete [] lhs_object;
	delete [] partition;
	delete [] alphabet;
	delete [] partition2;
	delete [] alphabet2;
	return 0;

}

int test_compare_cpu_gpu_2(){

	int *alphabet=new int[competition::ALPHABET];
	int *partition=new int [competition::NUM_RULES];

	int *alphabet2=new int[competition::ALPHABET];
	int *partition2=new int [competition::NUM_RULES];

	competition::reset_partition(partition,alphabet);
	competition::reset_partition(partition2,alphabet2);

	//Generate random rules
	int *rules_size = new int[competition::NUM_RULES+1];
	int total_lhs=competition::initialize_rules (rules_size, competition::NUM_RULES+1);

	int *lhs_object=new int[total_lhs];
	competition::initialize_lhs(lhs_object,total_lhs);

	std::cout<< "------ Testing GPU competition version 2 ------"<< std::endl;


	competition::print_header();
    competition::make_partition(partition,rules_size,lhs_object,total_lhs,alphabet);

	competition::make_partition_gpu(partition2,rules_size,lhs_object,total_lhs,alphabet2,true);

	competition::print_comparing_partition(partition,alphabet,partition2,alphabet2);
	competition::compare_partition(partition,alphabet,partition2,alphabet2);

	std::cout<< std::endl;

	delete [] rules_size;
	delete [] lhs_object;
	delete [] partition;
	delete [] alphabet;
	delete [] partition2;
	delete [] alphabet2;
	return 0;

}
int test_compare_cpu_gpu_random(){
	srand(time(NULL));

	int *alphabet=new int[competition::ALPHABET];
	int *partition=new int [competition::NUM_RULES];


	int *alphabet2=new int[competition::ALPHABET];
	int *partition2=new int [competition::NUM_RULES];

	std::cout<< "------ Testing GPU competition version 1 multiple ------"<< std::endl;
	competition::print_header();

	for(int i=0;i<NUM_COMPARISONS;i++){
		competition::reset_partition(partition,alphabet);
		competition::reset_partition(partition2,alphabet2);

		//Generate random rules
		int *rules_size = new int[competition::NUM_RULES+1];
		int total_lhs=competition::initialize_rules (rules_size, competition::NUM_RULES+1);

		int *lhs_object=new int[total_lhs];
		competition::initialize_lhs(lhs_object,total_lhs);

		competition::make_partition(partition,rules_size,lhs_object,total_lhs,alphabet);

		competition::make_partition_gpu(partition2,rules_size,lhs_object,total_lhs,alphabet2);

		int* normalized_cpu_partition=competition::normalize_partition(partition);
		int* normalized_gpu_partition=competition::normalize_partition(partition2);

		competition::compare_partition(normalized_cpu_partition,alphabet,normalized_gpu_partition,alphabet2);



		delete [] rules_size;
		delete [] lhs_object;

	}
	std::cout<< std::endl;

	delete [] partition;
	delete [] alphabet;
	delete [] partition2;
	delete [] alphabet2;
	return 0;

}
int test_compare_cpu_gpu_2_random(){
	srand(time(NULL));

	int *alphabet=new int[competition::ALPHABET];
	int *partition=new int [competition::NUM_RULES];


	int *alphabet2=new int[competition::ALPHABET];
	int *partition2=new int [competition::NUM_RULES];

	std::cout<< "------ Testing GPU competition version 2 multiple ------"<< std::endl;
	competition::print_header();

	for(int i=0;i<NUM_COMPARISONS;i++){
		competition::reset_partition(partition,alphabet);
		competition::reset_partition(partition2,alphabet2);

		//Generate random rules
		int *rules_size = new int[competition::NUM_RULES+1];
		int total_lhs=competition::initialize_rules (rules_size, competition::NUM_RULES+1);

		int *lhs_object=new int[total_lhs];
		competition::initialize_lhs(lhs_object,total_lhs);



		competition::make_partition(partition,rules_size,lhs_object,total_lhs,alphabet);

		competition::make_partition_gpu(partition2,rules_size,lhs_object,total_lhs,alphabet2,true);


		int* normalized_cpu_partition=competition::normalize_partition(partition);
		int* normalized_gpu_partition=competition::normalize_partition(partition2);


		competition::compare_partition(normalized_cpu_partition,alphabet,normalized_gpu_partition,alphabet2);



		delete [] rules_size;
		delete [] lhs_object;

	}

	std::cout<< std::endl;

	delete [] partition;
	delete [] alphabet;
	delete [] partition2;
	delete [] alphabet2;
	return 0;
}

int test_normalize_partition_2(){

	int *alphabet=new int[competition::ALPHABET];
	int *partition=new int [competition::NUM_RULES];

	int *alphabet2=new int[competition::ALPHABET];
	int *partition2=new int [competition::NUM_RULES];

	competition::reset_partition(partition,alphabet);
	competition::reset_partition(partition2,alphabet2);

	//Generate random rules
	int *rules_size = new int[competition::NUM_RULES+1];
	int total_lhs=competition::initialize_rules (rules_size, competition::NUM_RULES+1);

	int *lhs_object=new int[total_lhs];
	competition::initialize_lhs(lhs_object,total_lhs);

	std::cout<< "------ Testing CPU partition normalization against GPU competition version 2 ------"<< std::endl;


	competition::print_header();
	competition::print_rules(rules_size,lhs_object);


    competition::make_partition(partition,rules_size,lhs_object,total_lhs,alphabet);

	competition::make_partition_gpu(partition2,rules_size,lhs_object,total_lhs,alphabet2,true);
	int* normalized_cpu_partition=competition::normalize_partition(partition);
	int* normalized_gpu_partition=competition::normalize_partition(partition2);
	std::cout<< "------ Without normalization ------"<< std::endl;

	competition::print_comparing_partition(partition,alphabet,partition2,alphabet2);
	std::cout<< "------ With normalization ------"<< std::endl;
	competition::print_comparing_partition(normalized_cpu_partition,alphabet,normalized_gpu_partition,alphabet2);


	competition::compare_partition(normalized_cpu_partition,alphabet,normalized_gpu_partition,alphabet2);

	std::cout<< std::endl;

	delete [] normalized_gpu_partition;
	delete [] rules_size;
	delete [] lhs_object;
	delete [] partition;
	delete [] alphabet;
	delete [] partition2;
	delete [] alphabet2;
	return 0;

}

int test_normalize_partition(){

	int *alphabet=new int[competition::ALPHABET];
	int *partition=new int [competition::NUM_RULES];

	int *alphabet2=new int[competition::ALPHABET];
	int *partition2=new int [competition::NUM_RULES];

	competition::reset_partition(partition,alphabet);
	competition::reset_partition(partition2,alphabet2);

	//Generate random rules
	int *rules_size = new int[competition::NUM_RULES+1];
	int total_lhs=competition::initialize_rules (rules_size, competition::NUM_RULES+1);

	int *lhs_object=new int[total_lhs];
	competition::initialize_lhs(lhs_object,total_lhs);

	std::cout<< "------ Testing CPU partition normalization against GPU competition version 1 ------"<< std::endl;


	competition::print_header();
	competition::print_rules(rules_size,lhs_object);


    competition::make_partition(partition,rules_size,lhs_object,total_lhs,alphabet);

	competition::make_partition_gpu(partition2,rules_size,lhs_object,total_lhs,alphabet2);
	int* normalized_cpu_partition=competition::normalize_partition(partition);
	int* normalized_gpu_partition=competition::normalize_partition(partition2);
	std::cout<< "------ Without normalization ------"<< std::endl;

	competition::print_comparing_partition(partition,alphabet,partition2,alphabet2);
	std::cout<< "------ With normalization ------"<< std::endl;
	competition::print_comparing_partition(normalized_cpu_partition,alphabet,normalized_gpu_partition,alphabet2);


	competition::compare_partition(normalized_cpu_partition,alphabet,normalized_gpu_partition,alphabet2);

	std::cout<< std::endl;

	delete [] normalized_gpu_partition;
	delete [] rules_size;
	delete [] lhs_object;
	delete [] partition;
	delete [] alphabet;
	delete [] partition2;
	delete [] alphabet2;
	return 0;

}
