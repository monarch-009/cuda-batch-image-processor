# Makefile

NVCC       := nvcc
TARGET_DIR := bin
TARGET     := $(TARGET_DIR)/cuda_batch_image_processor
SRC_DIR    := src
INC_DIR    := include

SRCS       := $(SRC_DIR)/main.cu

NVCCFLAGS  := -O2 -std=c++14 -I$(INC_DIR)

all: $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -rf $(TARGET_DIR)

.PHONY: all clean
