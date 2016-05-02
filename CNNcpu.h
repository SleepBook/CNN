#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28
#define INPUT_CHANNEL 1

#define DEBUG_MODE 0

float non_linear(int type,float input);
//for type varible:
//	0 represent sigmoid
//	1 represent tanh
//	2 represent relu
int run_cpu();