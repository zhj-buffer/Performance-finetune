CC = gcc 
LD = gcc

SRCS = $(wildcard *.c)

OBJS = $(patsubst %c, %o, $(SRCS))

TARGET = bandwidth

.PHONY:all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(LD) -o $@ $^

 %.o:%.c
	$(CC) -Wall -march=native -std=gnu11 -O3 -c $^

clean:
	rm -f $(OBJS) $(TARGET)
