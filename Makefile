CC = cc
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
	CFLAGS =
	LDFLAGS = -lpng -framework OpenCL
else
	CFLAGS =  -I /usr/local/include/libpng ${AMDAPPSDKROOT}/include
	LDFLAGS = -L /opt/local/lib/ -L ${AMDAPPSDKROOT}/lib/x86_64 -lpng -lOpenCL
endif
SOURCES = main.c
OBJECTS = $(notdir $(SOURCES:.c=.o))
EXECUTE = main

all: $(OBJECTS) $(EXECUTE)

$(EXECUTE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -c

run:
	./$(EXECUTE) in.png out.png
clean:
	rm -rf *~ *.o $(EXECUTE)
