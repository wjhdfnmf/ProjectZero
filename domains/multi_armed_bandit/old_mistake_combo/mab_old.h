
/*

Multi-Armed Bandit
Copyright (c) 2016 William Gregory.  All rights reserved.

Player in a casino playing slots, tasked with finding the best paying slot.

ToDo: currently a constant reward.  add a zero reward and random reward with a bias.

*/

#ifndef _MULTI_ARMED_BANDIT_
#define _MULTI_ARMED_BANDIT_

#include <vector>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iostream>

#define MAB_DEBUG false
#define MAB_VERBOSE false

#define LYRAND (double)rand()/RAND_MAX

namespace MAB {

    //
    // Functions:
    // - move_greedy
    // - move_random
    //
    // - Player
    // - choose_slot
    // - add_reward
    // - return_cs
    //
    class Player {
    private:
        unsigned int slot_count;
        std::vector <double> current_standings;  // one per slot, learned reward
        unsigned int round;
        unsigned int last_move;
        unsigned int move_greedy() {
            if(MAB_DEBUG) std::cout << "DEBUG:Player:move_greedy" << std::endl;
            unsigned int t_move = 0;
            double t_best = current_standings.at(0);
            for (std::size_t i=1; i<current_standings.at(i); ++i) {
                if (current_standings.at(i) > t_best) {
                    t_move = i;
                    t_best = current_standings.at(i);
                }
            }
            last_move = 0;
            if(MAB_VERBOSE) std::cout << "VERBOSE:greedy slot:" << t_move << std::endl;
            return t_move;
        }
        unsigned int move_random() {
            if(MAB_DEBUG) std::cout << "DEBUG:Player:move_random" << std::endl;
            unsigned int t_move = rand() % slot_count;
            last_move = 1;
            if(MAB_VERBOSE) std::cout << "VERBOSE:random slot:" << t_move << std::endl;
            return t_move;
        }
    public:
        void setup(unsigned int in_c) {
            slot_count = in_c;
            round = 0;
            last_move = 0;
            for (std::size_t i=0; i<slot_count; ++i) {
                current_standings.push_back(1.0);
            }
        }
        unsigned int choose_slot() {
            if(MAB_DEBUG) std::cout << "DEBUG:Player:choose_slot" << std::endl;
            unsigned int t_slot;
            if (round < 10) t_slot = move_random();
            else {
                if (LYRAND > 0.6) t_slot = move_random();
                else t_slot = move_greedy();
            }
            round++;
            return t_slot;
        }
        void add_reward(unsigned int in_s, unsigned int in_r) {
            if(MAB_DEBUG) std::cout << "DEBUG:Player:add_reward" << std::endl;
            current_standings.at(in_s) = (current_standings.at(in_s) + in_r) / 2;
        }
        std::vector <double> return_cs() {
            return current_standings;
        }
        unsigned int return_last_move() {
            return last_move;
        }
    };

    //
    // Functions:
    // - Slot
    // - return_reward_bias
    // - generate_payout
    //
    class Slot {
    private:
        double reward;
    public:
        Slot(unsigned int in_max) {
            reward = rand() % in_max;
        }
        double return_reward_bias() { // DONT CHEAT!  LOG EYES ONLY
            return reward;
        }
        double generate_payout() {
            return reward; //* LYRAND;
        }
    };

    //
    // Functions:
    // - print_intro
    // - setup_player
    // - setup_slots
    // - pull_slot
    // - file_operations
    // 
    // - Casino
    // - cycle_games
    //
    class Casino {
    private:
        Player player;
        std::vector <Slot> slots;
        std::vector <double> slots_bias;
        unsigned int bias_max;
        unsigned int slot_count;
        unsigned int round;
        unsigned int max_round;
        void print_intro() {
            std::cout << "\nMulti-Armed Bandit\n" << std::endl;
            std::cout << "Rounds: " << max_round << std::endl;
            std::cout << "Slots: " << slot_count << std::endl;
            std::cout << "Max Bias: " << bias_max << std::endl;
            std::cout << "--------------------" << std::endl;
        }
        void setup_player() {
            player.setup(slot_count);
        }
        void setup_slots() {
            if(MAB_DEBUG) std::cout << "DEBUG:Casino:setup_slots" << std::endl;
            slots.clear();
            slots_bias.clear();
            for (std::size_t i=0; i<slot_count; ++i) {
                Slot s(bias_max);
                double t_bias = s.return_reward_bias();
                slots.push_back(s);
                slots_bias.push_back(t_bias);
            }
        }
        double pull_slot(unsigned int in) {
            if(MAB_DEBUG) std::cout << "DEBUG:Casino:pull_slot" << std::endl;
            if (in > (slots.size()-1)) return 0.0;  // unknown slot
            double t_payout = slots.at(in).generate_payout();
            return t_payout;
        }
        // in_move (0:greedy, 1:random)
        void file_operations(unsigned int in_move, unsigned int in_slot, std::vector <double> in_cs) {
            if(MAB_DEBUG) std::cout << "DEBUG:Casino:file_operations" << std::endl;
            std::ofstream file;
            if (round == 0) file.open("mab_results.txt", std::ios::out | std::ios::trunc);
            else file.open("mab_results.txt", std::ios::out | std::ios::app);
            if (!file.is_open()) std::cout << "UNHAPPY FILE" << std::endl;
            else {
                if (round == 0) {
                    for (std::size_t i=0; i<slots_bias.size(); ++i) {
                        if (i==0) file << "Slot: ";
                        if (i!=0) file << ", ";
                        file << i;
                    }
                    file << "\n";
                    for (std::size_t i=0; i<slots_bias.size(); ++i) {
                        if (i==0) file << "Bias: ";
                        if (i!=0) file << ", ";
                        file << slots_bias.at(i);
                    }
                    file << "\n\n";
                    file << "Info: round, move, slot, current_standings";
                }
                file << "\n";
                file << round << ", ";
                file << in_move << ", ";
                file << in_slot << ", ";
                for (std::size_t i=0; i<in_cs.size(); ++i) {
                    if (i!=0) file << ", ";
                    file << in_cs.at(i);
                }
                // current_standings
                file.close();
            }
        }
    public:
        Casino(unsigned int in_r, unsigned int in_c) {
            bias_max = 10; // in_r;
            slot_count = in_c;
            round = 0;
            max_round = 100;
            srand(time(NULL));
            setup_player();
            setup_slots();
        }
        void cycle_games() {
            print_intro();
            while (round < max_round) {
                if(MAB_DEBUG) std::cout << "DEBUG:Casino:cycle_games" << std::endl;
                std::vector <double> t_cs = player.return_cs();
                unsigned int t_slot = player.choose_slot();
                double t_reward = pull_slot(t_slot);
                player.add_reward(t_slot, t_reward);
                unsigned int t_move = player.return_last_move();
                file_operations(t_move, t_slot, t_cs);
                round++;
            }
        }
    };

}

#endif