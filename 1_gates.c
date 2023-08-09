#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define EPS 1e-3
#define ITERATIONS 1000 * 1000
#define LR_RATE 0.01f
#define MODEL(input1, input2, w1, w2, bias) input1 * w1 + input2 * w2 + bias

/*
Single neuron with 2 inputs
Modelling OR, AND, NAND
*/

typedef float sample[3];

// OR-gate
sample or_train[] = {
  //x1  x2 x1 + x2 = y 
    {0, 0, 0}, 
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

// AND-gate
sample and_train[] = {
  //x1  x2 x1 . x2 = y 
    {0, 0, 0}, 
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
};

// NAND-gate
sample nand_train[] = {
  //x1  x2 (x1 . x2)' = y 
    {0, 0, 1}, 
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

sample *train = or_train;

#define TRAIN_COUNT (float) 4

float sigmoidf(float x) {
    // Activation Function
    // Our output with all the weights and biases is unbounded
    // Therefore an activation function is used to get output in desired range
    // sig(x) = 1 / (1 + e^-x) has +1 and -1 as asymptotes as x -> inf and -inf resp.
    // so, for any real input x, output is (-1, 1).
    return 1.f / (1.f + expf(-x));
}

int pushf(float x) {
    if (x > 0.5f) {
        return 1;
    }
    return 0;
}

float cost(float w1, float w2, float bias) {
    float result = 0.0f;
    for (size_t i = 0; i < TRAIN_COUNT; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf((w1 * x1) + (w2 * x2) + bias) ;
        float diff = y - train[i][2];
        result += (diff * diff);
        
    }
    return result / TRAIN_COUNT; 
}

float float_random() {
    srand(time(NULL));
    return (float) rand() / (float) RAND_MAX; 
}

int main() {
    float w1 = float_random();
    float w2 = float_random();
    float bias = float_random();

    printf("MSE for randomised w1: %f, w2: %f and bias: %f is : %f\n", w1, w2, bias, cost(w1, w2, bias));


    for (size_t i = 0; i < ITERATIONS; ++i) {
        float dw_1 = (cost(w1 + EPS, w2, bias) - cost(w1, w2, bias)) / EPS;
        float dw_2 = (cost(w1, w2 + EPS, bias) - cost(w1, w2, bias)) / EPS;
        float db = (cost(w1, w2, bias + EPS) - cost(w1, w2, bias)) / EPS;

        w1 -= LR_RATE * dw_1;
        w2 -= LR_RATE * dw_2;
        bias -= LR_RATE * db;
    }

    printf("MSE for w1: %f and w2: %f (after grad. descent) and bias: %f is : %f\n", w1, w2, bias, cost(w1, w2, bias));

    float test[][3] = {
        {0, 0, -1}, 
        {0, 1, -1},
        {1, 0, -1},
        {1, 1, -1},
    };

    for (size_t i = 0; i < TRAIN_COUNT; i++) {
        test[i][2] = pushf(sigmoidf(MODEL(test[i][0], test[i][1], w1, w2, bias)));
    }
    printf("----------------------------------------\nOR GATE\n");

    for (size_t i = 0; i < TRAIN_COUNT; i++) {
        printf("%f OR %f = %f\n", test[i][0], test[i][1], test[i][2]);
    }
    

    return 0;
}