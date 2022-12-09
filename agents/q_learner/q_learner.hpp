
#include <vector>

#include "../../domains/gridspace/q_learner.hpp"

#define LYRAND (double)rand()/RAND_MAX

class QLearner {
private:
    /// variables
    Grid::Grid g;
    double learning_rate;
    double discount_factor;
    unsigned int q_table_x_size;
    unsigned int q_table_y_size;
    int loc_agent_x;  // delta?
    int loc_agent_y;
    int loc_target_x;
    int loc_target_y;
    unsigned int std::vector <std::vector <double> > q_table;
    /// functions
    std::vector <std::vector <double> > create_q_table() {
        std::vector <std::vector <double> > t_table;
        for (std::size_t i=0; i<q_table_x_size; i++) {
            std::vector <double> t_row;  // is it really a row?
            for (std::size_t j=0; j<q_table_y_size; j++) {
                t_row.push_back(LYRAND); // FIX: random value (0to1)?
            }
            t_table.push_back(t_row);
        }
        return t_table;
    }
    double calculate_value(unsigned int i, unsigned int j, double reward) {
        double old = q_table.at(i).at(j);
        // reward
        // optimal
        return old + learning_rate*(reward + discount_factor*optimal - old);
    }
    unsigned int decide_move(double val) {
        if (val < 0.25) return 1;
        else if (val < 0.5) return 2;
        else if (val < 0.75) return 3;
        return 4;
    }
public:
    /// variables
    /// functions
    QLearner() {
        learning_rate = 0.5;  // setting (0<x<1) (constant)
        discount_factor = 0.1;  // setting (start small)
        //
        q_table_x_size = 10;  // same as gridworld
        q_table_y_size = 10;
        //
        loc_agent_x = 0;
        loc_agent_y = 0;
        loc_target_x = 0;
        loc_target_y = 4;
    }
    void setup() {
        q_table = create_q_table();
        g.play_game(q_table_x_size, q_table_y_size, false);
    }
    void run() {
        while (true) {
            std::vector <double> state = g.give_state();
            unsigned int move = decide_move(state);=
            if (g.move_bot(move)) break;
            //// reward?
            calculate_value();
        }
    }
};
