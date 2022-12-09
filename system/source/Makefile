
# Makefile
# Ststem with Agent and Domain named "run"

# system files
SMAIN	=	main.cpp
SAGENT	=	network_trainer.hpp neural_network.hpp
SDOMAIN	=	pendulum.hpp pendulum.cpp
SSYSTEM	=	utilities.hpp system.hpp
SOUT	=	-o run

# flags
CXX 	=	g++
CVER	=	-std=c++11

# compile
all: $(system)

system: 
	$(CXX) $(CVER) $(SAGENT) $(SDOMAIN) $(SSYSTEM) $(SMAIN) $(SOUT)
