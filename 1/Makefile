
cflags=-Wno-write-strings

libs=-lGLEW -lGL -lGLU -lglut

dbgflags=-g

allobjs=framework.o Skeleton.o smoothtriangle.o

framework.o: framework.cpp framework.h
	g++ -c ${dbgflags} framework.cpp 
Skeleton.o: Skeleton.cpp
	g++ -c ${dbgflags} Skeleton.cpp ${cflags}

smoothtriangle.o: smoothtriangle.cpp
	g++ -c smoothtriangle.cpp

def: Skeleton.o framework.o
	g++ -o def framework.o Skeleton.o ${libs} ${cflags}

exa: smoothtriangle.o framework.o
	g++ -o exa framework.o smoothtriangle.o ${libs} ${cflags}

clean:
	rm def exa ${allobjs}