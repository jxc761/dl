CPP=g++





all: ../bin ../bin/cache_depth ../bin/detect ../bin/convert_depth


../bin:
	mkdir -p ../bin

# compile detect 
../bin/detect: detect.o
	$(CPP) detect.o -o $@


# compile convert 
../bin/convert_depth: convert.o
	$(CPP) convert.o -o $@

clean:
	rm -f *.o
	rm -f ../bin/detect
	rm -f ../bin/convert_depth
	rm -f ../bin/cache_depth



UNAME:=$(shell uname -s)
# compile flags on mac osx
ifeq ($(UNAME),Darwin)
	PROJECT_DIR=$(dir $(shell pwd) )
	CPPFLAGS:=$(CPPFLAGS) -c -Wall -D__forceinline=inline -std=c++11 -I../core -I../mac/include 
	LDFLAGS:=$(LDFLAGS) -L../mac/lib -lassimp -lembree -Wl,-rpath,${PROJECT_DIR}mac/lib
else
	PROJECT_DIR=$(dir $(shell pwd) )
	CPPFLAGS:=$(CPPFLAGS) -c -Wall -D__forceinline=inline -std=c++11 -I../core -I../linux/include 
	LDFLAGS:=$(LDFLAGS) -L../linux/lib -lassimp -lembree -Wl,-rpath,${PROJECT_DIR}linux/lib 
endif


TARGET=../bin/
CORE_SRC=$(wildcard ../core/*.cpp)
CORE_OBJ=$(CORE_SRC:.cpp=.o)
OBJS=$(CORE_OBJ) cache_depth.o depth_utils.o

../bin/cache_depth:$(OBJS)
	$(CPP) $(LDFLAGS) $(OBJS) -o $@
