
// Multi Armed Bandit - Example Run

#include "mab.h"

int main() {

    // (max bias, slot count)
    MAB::Casino mab(10, 10);
    
    unsigned int slot_to_pull = 1;

    mab.get_action(slot_to_pull);
    double payout = mab.give_reward();

    return 0;
}