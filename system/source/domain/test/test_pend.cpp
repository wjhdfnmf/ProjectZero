
/*
 * pendulum - test code
 *
 */

#include <iostream>

#include "../pendulum.hpp"

void delta_theta() {
	std::cout << "TEST: cart_balance_delta_theta" << std::endl;

	// add test P/F
	// FIX : last two failing

	// test 1, 359, 91, 89
	InvPend::Pendulum pend;
	std::cout << "0:90" << "\t" << pend.delta_theta(0*PI/180)*180/PI << std::endl;	// 90
	std::cout << "90:0" << "\t" << pend.delta_theta(90*PI/180)*180/PI << std::endl;	 // 0
	std::cout << "180:90" << "\t" << pend.delta_theta(180*PI/180)*180/PI << std::endl;	// 90
	std::cout << "270:180" << "\t" << pend.delta_theta(270*PI/180)*180/PI << std::endl;	 // 180
	std::cout << "360:90" << "\t" << pend.delta_theta(360*PI/180)*180/PI << std::endl;	// 90
	//
	std::cout << "cart_balance_delta_theta - done" << std::endl;
}

int main() {

	delta_theta();
	
	return 0;
}
