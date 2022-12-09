
#ifndef _UTILITIES_
#define _UTILITIES_

#include <string>
#include <fstream>
#include <iostream>
#include <vector>

class Util {

public:

	// display progress bar
	// input: progress
	double progress_display(double in)
	{
		unsigned int update_interval = 100.0;  // setting
		unsigned int bar_width = 40;  // setting
		unsigned int t_progress = int(in*10000);
		if (t_progress % update_interval == 0) {
			std::cout << "[";
			unsigned int bar_pos = bar_width * in;
			unsigned int bar_pos_em = bar_width - bar_pos;
			for (std::size_t i=1; i<=bar_pos; ++i) {
				if (i < bar_pos) std::cout << "=";
				else std::cout << ">";
			}
			for (std::size_t i=0; i<bar_pos_em; ++i) {
				std::cout << " ";
			}
			std::cout << "] " << int(in * 100) << "%" << "\r";
			std::cout.flush();
		}
	}

	// export vector to file
	// input: filename, vector of doubles to export
	void export_vector_vector(std::string in_fn, std::vector <std::vector <double> > in_msg) {
		std::ofstream file;
		file.open(in_fn.c_str(), std::ofstream::out | std::ofstream::trunc);
		for (std::size_t i=0; i<in_msg.size(); ++i) {
			for (std::size_t j=0; j<in_msg.at(i).size(); ++j) {
				if (j!=0) file << ", ";
				file << in_msg.at(i).at(j);
			}
			if (i!=in_msg.size()) file << "\n";
		}
		file.close();
	}

	// seconds to minutes
	// output: <min, sec>
	std::vector <unsigned int> time_convert(unsigned int in_dt) {
		std::vector <unsigned int> t;
		if (in_dt > 60) {
			unsigned int t_min = (unsigned int)(in_dt/60);
			t.push_back(t_min);
			t.push_back(in_dt-60*t_min);
		} else {
			t.push_back(0);
			t.push_back(in_dt);
		}
		return t;
	}
};

#endif