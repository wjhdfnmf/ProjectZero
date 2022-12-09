
/*

Multi-Armed Bandit (Domain)
Copyright (c) 2016 William Gregory.  All rights reserved.

INFO: Casino full of slot machines.  Choose one and get a reward.  Each slot has a random bias less than set max.

THOUGHS:
    - Check get_action.  Best way to recieve action?  Assume int for slot number in first slot or a range from 0 to 1 of a double coresponding to different slots?
    - Does this domain have a state for agent?  Number of slots? I guess it would be the slots bias but that is cheating.

*/

#ifndef _MULTI_ARMED_BANDIT_
#define _MULTI_ARMED_BANDIT_

#include <vector>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iostream>

//#define MAB_DEBUG false
//#define MAB_VERBOSE false

#ifndef LYRAND
#define LYRAND (double)rand()/RAND_MAX
#endif

namespace MAB {

    //
    // Functions:
    // - Slot
    // - return_reward_bias
    // - generate_payout
    //
    class Slot {
    private:
        double max_reward;
    public:
        Slot(unsigned int in_max) {
            max_reward = in_max;
        }
        double return_reward_bias() { // DONT CHEAT!
            return max_reward;
        }
        double generate_payout() {
            double t;
            if (LYRAND > 0.8) t = 0.0;  // zero reward (20% chance)
            else t = max_reward * LYRAND;  // max reward * LYRAND (80% chance)
            return t;
        }
    };

    //
    // Functions:
    // - return_slot_count
    // - pull_slot (give slot, recieve reward)
    // ---
    // - Casino (setup)
    // - give_state
    // - give_reward
    // - get_action
    //
    class Casino {
    private:
        std::vector <Slot> slots;
        double last_reward;
        // return number of slots in casino
        unsigned int return_slot_count() {
            unsigned int t;
            t = slots.size();
            return t;
        }
        // select a slot and recieve a reward
        double pull_slot(unsigned int& in) {
            if (in > (slots.size()-1)) return -1.0;  // unknown slot
            return slots.at(in).generate_payout();  // payout
        }
    public:
        Casino(unsigned int in_b, unsigned int in_c) {
            srand(time(NULL));
            // setup slots
            slots.clear();
            last_reward = -1.0;
            for (std::size_t i=0; i<in_c; ++i) {
                Slot s(in_b * LYRAND);
                slots.push_back(s);
            }
        }
        // return MAB state
        std::vector <double> give_state(void) {
            std::vector <double> t;
            double t2 = (double) return_slot_count();  // state?
            t.push_back(t2);  // state?
            return t;
        }
        // return last reward
        std::vector <double> give_reward(void) {
            std::vector <double> t;
            t.push_back(last_reward);
            last_reward = -1.0;  // reset reward
            return t;
        }
        // recieve action and pull slot
        bool get_action(std::vector <double> in) {
            unsigned int t = (unsigned int) in.at(0);
            last_reward = pull_slot(t);
            if (last_reward == -1.0) return false;
            else return true;
        }
    };

}

#endif