CC = g++
CFLAGS = -std=c++11
TARGET = main
SRCS = main.cpp Layer.cpp Network.cpp Activations.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

