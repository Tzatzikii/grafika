
cflags=-Wno-write-strings

libs=-lGLEW -lGL -lGLU -lglut

dbgflags=-g

allobjs=framework.o Skeleton.o Skeleton2.o

main: Skeleton.o framework.o
	g++ -o main framework.o Skeleton.o ${libs} ${cflags}
framework.o: framework.cpp framework.h
	g++ -c ${dbgflags} framework.cpp 
Skeleton.o: Skeleton.cpp
	g++ -c ${dbgflags} Skeleton.cpp ${cflags}

Skeleton2.o: Skeleton2.cpp
	g++ -c ${dbgflags} Skeleton2.cpp ${cflags}

Skeleton2: Skeleton2.o framework.o
	g++ -o example framework.o Skeleton2.o ${libs} ${cflags}

clean:
	rm main ${allobjs}