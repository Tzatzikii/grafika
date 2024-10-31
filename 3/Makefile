
cflags=-Wno-write-strings

libs=-lGLEW -lGL -lGLU -lglut

dbgflags=-g

allobjs=framework.o Skeleton.o

main: Skeleton.o framework.o
	g++ -o main framework.o Skeleton.o ${libs} ${cflags}
framework.o: framework.cpp framework.h
	g++ -c ${dbgflags} framework.cpp 
Skeleton.o: Skeleton.cpp
	g++ -c ${dbgflags} Skeleton.cpp ${cflags}



clean:
	rm main ${allobjs}