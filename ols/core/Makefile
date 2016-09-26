CPP=g++

all: ols.o basic_io.o

%.o: %.cpp
	$(CPP) -c -Wall -std=c++11 $< -o $@

clean:
	rm -f *.o
