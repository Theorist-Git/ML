#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define EPS 1e-3
#define ITERATIONS 1000 * 1000
#define LR_RATE 0.01f
#define MODEL(input1, input2, w1, w2, bias) input1 * w1 + input2 * w2 + bias

/*
XOR and XNOR cannot be implemented with a single neuron
But they can be represented by AND, OR and NAND which are all 
modellable by a single neuron.

We know (x + y).(x.y)' == x ^ y, therefore its not absurd to think
that given enough perceptrons, we can model what is a "combination" of 
operations modellable by single neurons. Therefore, we add 2 layers

1st: L1: one that with perform OR and NAND ops#. 

L1 will generate inputs for L2
2nd: L2: Performs AND op# and outputs a number between 0 and 1. For a well fitted model
         values will be very close to 0 and 1.   

#(at least we model it this way, might not be true for the actual model obtained)
*/

typedef struct {
    // 3 neurons
    // 2 layers

    // Layer 1: OR and NAND
    float or_w1;
    float or_w2;
    float or_bias;

    float nand_w1;
    float nand_w2;
    float nand_bias;

    // Layer 2: AND
    float and_w1;
    float and_w2;
    float and_bias;

    // x ^ y = (x + y).(x.y)' = x.y' + x'.y
} xor;

typedef float sample[3];
#define TRAIN_COUNT (float) 4

// XOR-gate
sample xor_train[] = {
    {0, 0, 0}, 
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

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

// XNOR-gate
sample xnor_train[] = {
    {0, 0, 1}, 
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
};

sample* train = xor_train;

float sigmoidf(float x) {
    // Activation Function
    // Our output with all the weights and biases is unbounded
    // Therefore an activation function is used to get output in desired range
    // sig(x) = 1 / (1 + e^-x) has +1 and -1 as asymptotes as x -> inf and -inf resp.
    // so, for any real input x, output is (-1, 1).

    return 1.f / (1.f + expf(-x));
}

float forward(xor m, float x1, float x2) {

    // x1 and x2 are input values from input layer
    // Computes output of L1 for each of the 2 neurons and passes them as the 2
    // inputs to L2
    float l1_or = sigmoidf((x1 * m.or_w1) + (x2 * m.or_w2) + (m.or_bias));
    float l1_nand = sigmoidf((x1 * m.nand_w1) + (x2 * m.nand_w2) + (m.nand_bias));

    // now l1_or and l1_nand become x1 and x2
    // Computes the output of NN.
    float l2_and = sigmoidf((l1_or * m.and_w1) + (l1_nand * m.and_w2) + m.and_bias);

    return l2_and;
}

float cost(xor m) {
    float result = 0.0f;
    for (size_t i = 0; i < TRAIN_COUNT; ++i) {
        float x1 = train[i][0];
        float x2 = train[i][1];

        // We directly get output values of L2 i.e the NN with forward func(see its comments).
        float y = forward(m, x1, x2);
        float diff = y - train[i][2];
        result += (diff * diff);
    }

    return result / TRAIN_COUNT;
}

xor finite_difference(xor m) {
    xor g; // All the parameters of g will store their approx dertivatives at the end
           // i.e g.or_w1 will store d_or_w1 (~ ∂C / ∂or_w1)
           // xor m remains unchanged 
    float c = cost(m);
    float saved;

    saved = m.or_w1;
    m.or_w1 += EPS;
    g.or_w1 = (cost(m) - c)/EPS;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += EPS;
    g.or_w2 = (cost(m) - c)/EPS;
    m.or_w2 = saved;

    saved = m.or_bias;
    m.or_bias += EPS;
    g.or_bias = (cost(m) - c)/EPS;
    m.or_bias = saved;

    saved = m.nand_w1;
    m.nand_w1 += EPS;
    g.nand_w1 = (cost(m) - c)/EPS;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += EPS;
    g.nand_w2 = (cost(m) - c)/EPS;
    m.nand_w2 = saved;

    saved = m.nand_bias;
    m.nand_bias += EPS;
    g.nand_bias = (cost(m) - c)/EPS;
    m.nand_bias = saved;

    saved = m.and_w1;
    m.and_w1 += EPS;
    g.and_w1 = (cost(m) - c)/EPS;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += EPS;
    g.and_w2 = (cost(m) - c)/EPS;
    m.and_w2 = saved;

    saved = m.and_bias;
    m.and_bias += EPS;
    g.and_bias = (cost(m) - c)/EPS;
    m.and_bias = saved;

    return g;
}

xor learn(xor m, xor g) {
    // Move all derivatives opposite to their ∇C or in the direction of -∇C.
    // LR_RATE gives the size of the step taken in direction of -∇C.

    m.or_w1 -= LR_RATE *  g.or_w1;
    m.or_w2 -= LR_RATE *  g.or_w2;
    m.or_bias -= LR_RATE *  g.or_bias;
    m.nand_w1 -= LR_RATE *  g.nand_w1;
    m.nand_w2 -= LR_RATE *  g.nand_w2;
    m.nand_bias -= LR_RATE *  g.nand_bias;
    m.and_w1 -= LR_RATE *  g.and_w1;
    m.and_w2 -= LR_RATE *  g.and_w2;
    m.and_bias -= LR_RATE *  g.and_bias;

    return m;
}

float float_random() {
    return (float) rand() / (float) RAND_MAX; 
}

xor rand_xor(void) {
    xor m;

    m.or_w1 = float_random();
    m.or_w2 = float_random();
    m.or_bias = float_random();
    m.nand_w1 = float_random();
    m.nand_w2 = float_random();
    m.nand_bias = float_random();
    m.and_w1 = float_random();
    m.and_w2 = float_random();
    m.and_bias = float_random();

    return m;
}

void print_xor(xor m) {

    printf("or_w1 = %f\n", m.or_w1);
    printf("or_w2 = %f\n", m.or_w2);
    printf("or_bias = %f\n", m.or_bias);
    printf("nand_w1 = %f\n", m.nand_w1);
    printf("nand_w2 = %f\n", m.nand_w2);
    printf("nand_bias = %f\n", m.nand_bias);
    printf("and_w1 = %f\n", m.and_w1);
    printf("and_w2 = %f\n", m.and_w2);
    printf("and_bias = %f\n", m.and_bias);

}

int main() {
    srand(time(0));
    xor m = rand_xor();
    print_xor(m);

    printf("Cost before learning: %f\n", cost(m));

    for (size_t i = 0; i < ITERATIONS; i++) {
        xor g = finite_difference(m);
        m = learn(m, g);
    }
    print_xor(m);
    printf("Cost after learning: %f\n", cost(m));
    printf("\n\n");

    printf("------------------------------\n");
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            printf("%zu ^ %zu = %f\n", i, j, forward(m, i, j));
        }
    }
    printf("------------------------------\n");
    printf("\"OR\" neuron:\n");
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            printf("%zu | %zu = %f\n", i, j, sigmoidf(m.or_w1*i + m.or_w2*j + m.or_bias));
        }
    }
    printf("------------------------------\n");
    printf("\"NAND\" neuron:\n");
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            printf("~(%zu & %zu) = %f\n", i, j, sigmoidf(m.nand_w1*i + m.nand_w2*j + m.nand_bias));
        }
    }
    printf("------------------------------\n");
    printf("\"AND\" neuron:\n");
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            printf("%zu & %zu = %f\n", i, j, sigmoidf(m.and_w1*i + m.and_w2*j + m.and_bias));
        }
    }



    return 0;
}