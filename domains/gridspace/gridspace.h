
//
// Gridspace
// William - September 13th 2016
//

// Idea: create grid stuct

// Info:
// grid.at(row).at(column)
// grid.at(y).at(x)

// ToDo:
// complete bot player option
// needs scaling, add to network?

#include <iostream> // cout, endl
#include <vector>   // vector
#include <string>   // string
#include <cstdlib>  // rand, srand, hmm
#include <ctime>	// time

//#include "../../abstract_connection/domain.h"

///////////////////////////////////////////////////////////////
class Grid
{
private:
	//// variables
	// error when true
	bool runtime_error;
	// true when playing game, fale to exit
	bool play;
	// game score
	int score;
	// like score (reset each level)
	int targets_hit;
	// game level
	int level;
	// player type
	bool human_player;
	bool bot_display_grid;
	// grid size
	int grid_x;
	int grid_y;
	// target position(s)
	std::vector <int> target_x;
	std::vector <int> target_y;
	// target counts
	int targets_total;
	// player starting position and after, current position (updated)
	int player_x;
	int player_y;
	// display settings
	bool display_include_space;
	bool display_clear_screen;
	// grid value settings
	int grid_val_null;
	int grid_val_target;
	int grid_val_player;
	int move_n;
	int move_e;
	int move_s;
	int move_w;
	std::string grid_sym_null;
	std::string grid_sym_target;
	std::string grid_sym_player;
	// grid
	std::vector <std::vector <int> > grid;
	
	//// functions
	void error_call(std::string);
	void debug_call(std::string);
	void exit_game();
	void display_help();
	// generate grid
	std::vector <std::vector <int> > generate_grid(int, int);
	// generate targets
	void generate_targets(int);
	// place grid target
	std::vector <std::vector <int> > place_grid_target(std::vector <std::vector <int> >, int, int);
	// place grid player
	std::vector <std::vector <int> > place_grid_player(std::vector <std::vector <int> >, int, int);
	// display grid
	void display_grid(std::vector <std::vector <int> >);
	// check legal move
	bool check_legal_move(int);
	// human get move
	int human_get_move();
	// comp get move
	void get_action(std::vector <double>);
	std::vector <double> give_state();
	// apply move
	std::vector <std::vector <int> > apply_move(std::vector <std::vector <int> >, int);
	// check target reached
	bool check_target_reached();
	// main run human
	void main_run_human();
	// main run bot
	void main_run_bot();
public:
	Grid();
	void play_game(int, int, bool);
};

Grid::Grid()
{
	// - settings -
	// player position (updated)
	player_x = 0;
	player_y = 0;
	// display optioins
	bot_display_grid = true;
	display_include_space = true;
	display_clear_screen = true;
	//
	grid_sym_null = "-";
	grid_sym_target = "x";
	grid_sym_player = "o";
	// - end settings -

	// - do not modify -
	play = true;
	runtime_error = false;
	target_x.clear();
	target_y.clear();
	score = 0;
	targets_hit = 0;
	level = 1;
	grid_val_null = 0;
	grid_val_target = 1;
	grid_val_player = 2;
	move_n = 1;
	move_e = 2;
	move_s = 3;
	move_w = 4;
	// - end do not modify -
}

// .done
// Error handler
//
void Grid::error_call(std::string in_message)
{
	std::cout << "- error: " << in_message << std::endl;
	runtime_error = true;
}

void Grid::debug_call(std::string in_message)
{
	if (debug) std::cout << "- debug: " << in_message << std::endl;
}

void Grid::exit_game()
{
	system("clear");
	std::cout << "\nfinal score: " << score << std::endl;
	std::cout << "good day\n" << std::endl;
	std::exit(0);
}

void Grid::display_help()
{
	std::cout << "\n\t - HELP -" << std::endl;
	std::cout << "\nTo move:\n\n";
	std::cout << " - w,d,s,a\n - n,s,e,w\nproceded by enter key";
	std::cout << "\n\nOther commands:\n\n";
	std::cout << " - exit\n - reprint\n - help\n - auto_win\n";
	std::cout << std::endl;
}

//
// Generate grid given size_x and size_y
// where x is the column size and y is the row size
//
std::vector <std::vector <int> > Grid::generate_grid(int in_size_x, int in_size_y)
{
	std::vector <std::vector <int> > t_grid;
	for (int r=0; r<in_size_y; ++r)
	{
		std::vector <int> t_column;
		for (int c=0; c<in_size_x; ++c)
		{
			t_column.push_back(grid_val_null);
		}
		t_grid.push_back(t_column);
	}
	return t_grid;
}

//
// Generate targets, count based on input number
//
void Grid::generate_targets(int in_num)
{
	target_x.clear();
	target_y.clear();
	for (int i=0; i<in_num; ++i)
	{
		int x = rand() % (grid_x-1);
		int y = rand() % (grid_y-1);
		while (x==player_x && y==player_y) x = rand() % grid_x-1;
		target_x.push_back(x);
		target_y.push_back(y);
		if (verbose) std::cout << "target: " << x << ", " << y << std::endl;
	}
}

//
// Place a target in the given grid at the given position and return the grid
//
std::vector <std::vector <int> > Grid::place_grid_target(std::vector <std::vector <int> > in_grid, int in_x, int in_y)
{
	debug_call("place_grid_target()");
	in_grid.at(in_y).at(in_x) = grid_val_target;
	return in_grid;
}

//
// Place player on grid (only first time, might change this)
//
std::vector <std::vector <int> > Grid::place_grid_player(std::vector <std::vector <int> > in_grid, int in_x, int in_y)
{
	debug_call("place_grid_player()");
	in_grid.at(in_y).at(in_x) = grid_val_player;
	return in_grid;
}

//
// Display given grid
//
void Grid::display_grid(std::vector <std::vector <int> > in_grid)
{
	debug_call("display_grid()");
	if (display_clear_screen && !debug) system("clear");
	std::cout << "Grid Space: " << std::endl << std::endl;
	std::cout << " - score: " << score << std::endl;
	std::cout << " - targets total: " << level << std::endl;
	std::cout << " - targets remaining: " << level-targets_hit << std::endl;
	std::cout << std::endl;
	for (int r=0; r<in_grid.size(); ++r)
	{
		for (int c=0; c<in_grid.at(r).size(); ++c)
		{
			if (display_include_space) std::cout << " ";
			if (in_grid.at(r).at(c) == grid_val_null)
				std::cout << grid_sym_null;
			else if (in_grid.at(r).at(c) == grid_val_target)
				std::cout << grid_sym_target;
			else if (in_grid.at(r).at(c) == grid_val_player)
				std::cout << grid_sym_player;
			else
				std::cout << " ?";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	if (level == 1) std::cout << "type 'help' for help\n" << std::endl;
	debug_call("display_grid() end");
}

// .done.test.sub
// Check if selected move is legal
//
bool Grid::check_legal_move(int in_move)
{
	debug_call("check_legal_move() start");
	bool t_legal = true;
	if (in_move == move_n) // north
	{
		if (player_y-1 < 0) t_legal = false;
	}
	else if (in_move == move_e) // east
	{
		if (player_x+1 > grid_x-1) t_legal = false;
	}
	else if (in_move == move_s) // south
	{
		if (player_y+1 > grid_y-1) t_legal = false;
	}
	else if (in_move == move_w) // west
	{
		if (player_x-1 < 0) t_legal = false;
	}
	else error_call("check_legal_move() input error");
	debug_call("check_legal_move() end");
	return t_legal;
}

// .done.test.main
// Get move from player and confirm legal move
//
int Grid::human_get_move()
{
	debug_call("human_get_move()");
	int t_move;
	bool choose_move = true;
	while (choose_move)
	{
		std::string t_move_input;
		while (true) // user input
		{
			std::cout << "move options: n, s, e, w" << std::endl;
			std::cout << "enter move: ";
			std::getline(std::cin, t_move_input);
			if (std::cin.fail())
			{
				std::cin.clear();
				std::cin.ignore(256, 'n');
				//error_call("human_get_move() input fail")
				std::cout << " - error with input" << std::endl << std::endl;
			} else break;
		}
		bool nope = false;
		// map input to move options
		if (t_move_input == "N" || t_move_input == "n" || t_move_input == "w")
			t_move = move_n; // north
		else if(t_move_input == "E" || t_move_input == "e" || t_move_input == "d")
			t_move = move_e; // south
		else if (t_move_input == "S" || t_move_input == "s" || t_move_input == "s")
			t_move = move_s; // east
		else if (t_move_input == "W" || t_move_input == "w" || t_move_input == "a")
			t_move = move_w; // west
		else if (t_move_input == "exit")
			exit_game();
		else if (t_move_input == "help")
		{
			display_help();
			nope = true;
		}
		else if (t_move_input == "auto_win")
		{
			std::cout << "\nlol you wish\n" << std::endl;
			nope = true;
		}
		else if (t_move_input == "reprint")
		{
			nope = true;
			display_grid(grid);
		}
		else
		{
			std::cout << " - that is an unrecognised move" << std::endl << std::endl;
			nope = true;
		}
		if (!nope)
		{
			// check legal
			if (check_legal_move(t_move)) choose_move = false;
			else std::cout << " - that is an invalid move due to boundry" << std::endl << std::endl;
		}
	}
	debug_call("human_get_move() end");
	return t_move;
}

//
// Get move from computer (function call)
//
void Grid::get_action(std::vector <double> in)
{
#ifdef GS_DEBUG
	debug_call("get_action()");
#endif
	if (in.size() == 4) {
		unsigned int t_max_pos = 0;
		double t_max_val = in.at(0);
		// find max value
		for (std::size_t i=0; i<4; ++i) {
			if (in.at(i) > t_max_val) {
				t_max_pos = i;
				t_max_val = in.at(i);
			}
		}
		// set move
		switch (t_max_pos) {
			case 0:
				t_move = move_n;
				break;
			case 1:
				t_move = move_s;
				break;
			case 2:
				t_move = move_e;
				break;
			default:
				t_move = move_w;
		}
	} else {
		error_call("get_action() input fail");
	}
}

//
// Return dx, dy from player to target
//
std::vector <double> Grid::give_state() {
#ifdef GS_DEBUG
	debug_call("give_state()");
#endif
	std::vector <double> t_out;
	int t_x = player_x - target_x.at(0);  // only one target
	int t_y = player_y - target_y.at(0);
	t_out.push_back(t_x);
	t_out.push_back(t_y);
	return t_out;
}

//
// Apply the given move
//
std::vector <std::vector <int> > Grid::apply_move(std::vector <std::vector <int> > in_grid, int in_move)
{
#ifdef GS_DEBUG
	debug_call("apply_move()");
#endif
	int new_player_x = player_x;
	int new_player_y = player_y;
	if (in_move == move_n)
		new_player_y--;
	else if (in_move == move_e)
		new_player_x++;
	else if (in_move == move_s)
		new_player_y++;
	else if (in_move == move_w)
		new_player_x--;
	else error_call("apply_move() movement unknown");
	// remove player from old position
	in_grid.at(player_y).at(player_x) = grid_val_null;
	// add player to new position
	in_grid.at(new_player_y).at(new_player_x) = grid_val_player;
	// set new player position values
	player_x = new_player_x;
	player_y = new_player_y;
	// return
	return in_grid;
}

//
// Check target reached by player
//
bool Grid::check_target_reached()
{
#ifdef GS_DEBUG
	debug_call("check_target_reached() start");
#endif
	for (int i=0; i<target_x.size(); ++i)
	{
		if (player_x == target_x.at(i))
		{
			if (player_y == target_y.at(i)) 
			{
				target_x.erase(target_x.begin() + i);
				target_y.erase(target_y.begin() + i);
				++score;
				++targets_hit;
			}
		}
	}
	bool t_all_targets_hit = false;
	if (target_x.size() == 0)
	{
		t_all_targets_hit = true;
		++level;
	}
	return t_all_targets_hit;
}

//
// Main game loop for human
//
void Grid::main_run_human()
{
#ifdef GS_DEBUG
	debug_call("main_run_human() start");
#endif
	// generate grid
	grid = generate_grid(grid_x, grid_y);
	// generate targets
	generate_targets(level);
	// place targets
	for (int i=0; i<target_x.size(); ++i)
	{
		grid = place_grid_target(grid, target_x.at(i), target_y.at(i));
	}
	// place player
	grid = place_grid_player(grid, player_x, player_y);
	// cycle game
	while(play && !runtime_error)
	{
		display_grid(grid);
		int move = human_get_move();
		grid = apply_move(grid, move);
		if (check_target_reached())
		{
			// next round
			targets_hit = 0;
			generate_targets(level);
			for (int i=0; i<target_x.size(); ++i)
			{
				grid = place_grid_target(grid, target_x.at(i), target_y.at(i));
			}
		}
	}
}

void Grid::setup_game_bot()
{
	unsigned int t_targets = 1;  // target count (const)
	// generate grid
	grid = generate_grid(grid_x, grid_y);
	// generate targets
	generate_targets(t_targets);
	// place targets
	for (int i=0; i<target_x.size(); ++i)
	{
		grid = place_grid_target(grid, target_x.at(i), target_y.at(i));
	}
	// place player
	grid = place_grid_player(grid, player_x, player_y);
	/*
	while(play && !runtime_error)
	{
		if (bot_display_grid) display_grid(grid);
		//give_state();  // trainer should call this
		//unsigned int t_move = get_action(); // domain.h
		grid = apply_move(grid, t_move);
		if (check_target_reached())
		{
			// next round
			targets_hit = 0;
			generate_targets(t_targets);
			for (int i=0; i<target_x.size(); ++i)
			{
				grid = place_grid_target(grid, target_x.at(i), target_y.at(i));
			}
		}
	}
	*/
}

bool Grid::move_bot(unsigned int move)
{
	grid = apply_move(grid, move);
	return check_target_reached();
}

void Grid::play_game(int in_x, int in_y, bool in_pt)
{
	std::srand(std::time(0));
	human_player = in_pt;
	grid_x = in_x;
	grid_y = in_y;
	// start game
	//if (human_player) main_run_human();
	//else main_run_bot();
	setup_game_bot();
}

