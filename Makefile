CC = clang
CFLAGS =  -I /usr/include/libpng -I ${AMDAPPSDKROOT}/include

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
	LDFLAGS = -L /opt/local/lib/ -lpng -framework OpenCL
else
	LDFLAGS = -L /opt/local/lib/ -L ${AMDAPPSDKROOT}/lib/x86_64 -lpng -lOpenCL
endif
#LDFLAGS = -L /opt/local/lib/ -lpng -framework OpenCL
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
