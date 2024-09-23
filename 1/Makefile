
cflags=-Wno-write-strings

libs=-lGLEW -lGL -lGLU -lglut

allobjs=framework.o Skeleton.o smoothtriangle.o

framework.o: framework.cpp framework.h
	g++ -c framework.cpp

Skeleton.o: Skeleton.cpp
	g++ -c Skeleton.cpp

smoothtriangle.o: smoothtriangle.cpp
	g++ -c smoothtriangle.cpp

def: Skeleton.o framework.o
	g++ -o def framework.o Skeleton.o ${libs} ${cflags}

exa: smoothtriangle.o framework.o
	g++ -o exa framework.o smoothtriangle.o ${libs} ${cflags}

clean:
	rm def exa ${allobjs}