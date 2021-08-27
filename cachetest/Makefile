CC = cc
LD = cc
SRCS = $(wildcard *.c)
OBJS = $(patsubst %c, %o, $(SRCS))
INCLUDE = 
LIB =
CFLAGS = -Wall -O3 -std=gnu11 -DBLOCK_EXP=1
LDFLAGS = 

TARGETLIST = $(patsubst %.c, %, $(SRCS))

.PHONY:all clean

all: $(TARGETLIST)

#$(TARGETLIST): $(OBJS)
#	$(LD) $(LIB) $(LDFLAGS) -o $@ $^
%.o:%.c
	$(CC) $(CFLAGS) -c $^
	$(LD) $(LIB) $(LDFLAGS) -o $@ $^
 	 
clean:
	rm -f $(OBJS) $(TARGETLIST)
