#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define LR_RATE 0.01f
#define ITERATIONS 200
#define EPS 1e-6
#define MODEL(input, weight, bias) (input * weight + bias)
#define TRAIN_COUNT (float)(sizeof(train) / sizeof(train[0]))

/*
Single neuron with a single input.
Simple linear regression also implemented.
Switched out gradient descent for finite differences model (courtesy tscoding)
*/

        // x  y
float train[][2] = {
    {0, 0}, 
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

float mean(float train[][2], int op, size_t x) {
    // op = 0 => mean of [i][0] (x)
    // op = 1 => mean of [i][1] (y)
    // op = 2 => mean of ([i][0]) ^ 2   (x^2)
    // op = 3 => mean of ([i][1]) ^ 2   (y^2)
    // op = 4 => mean of ([i][0] * [i][1])  (x * y)
    float sum = 0;
    switch (op){
        case 0:
            for (size_t i = 0; i < x; i++) {
                sum += train[i][0];
            }
            break;

        case 1:
            for (size_t i = 0; i < x; i++) {
                sum += train[i][1];
            }
            break;
        
        case 2:
            for (size_t i = 0; i < x; i++) {
                sum += (train[i][0] * train[i][0]);
            }
            break;
        
        case 3:
            for (size_t i = 0; i < x; i++) {
                sum += (train[i][1] * train[i][1]);
            }
            break;

        case 4:
            for (size_t i = 0; i < x; i++) {
                sum += (train[i][0] * train[i][1]);
            }
            break;

        default:
            break;
    }

    return sum / x;

}

// y = x * w + bias
// x : input
// y : output
// w : parameter, changing this, changes the data our model outputs

float float_random() {
    srand(time(NULL));
    return (float) rand() / (float) RAND_MAX; 
}

// x1, x2, x3, ... -> input variables, b -> bias
// w1, w2, w3, ... -> weights of corresponding paramter
// y = x1*w1 + x2*w2 + x3*w3 + ... + b

// In this example we have a single neuron with a single input x
// weight(parameter) w and output y

float cost(float parameter, float bias) {
    // We take Mean Squared error as it is positive and ampilifies errors 
    // between actual and predicted values.

    // This is our cost function
    // Measures the cost of our model
    float result = 0.0f;
    for (size_t i = 0; i < TRAIN_COUNT; ++i) {
        float x = train[i][0];
        float y = x * parameter + bias;
        float diff = y - train[i][1];
        result += (diff * diff) ; 
    }

    result /= TRAIN_COUNT;

    return result;
}


int main() {
    // INIT random values for w and bias
    float w = 0;
    float b = 0;

    // Exact values of w and b for ∂C(w, b) / ∂w = 0 and ∂C(w, b) / ∂b = 0
    // This method is not a computationally feasable way of finding minima of our cost function.
    // This will be very slow for large datasets and for more than 1 parameter
    double w_exact = (mean(train, 4, 5) - mean(train, 0, 5) * mean(train, 1, 5)) / \
                                                                           \
                      (mean(train, 2, 5) - (mean(train, 0, 5) * mean(train, 0, 5)));

    double b_exact = mean(train, 1, 5) - w_exact * mean(train, 0, 5);

    printf("(EXACT)Predicted value of y(for x = %f) = %f v/s real value: %f\n", 5.0f, MODEL(5, w_exact, b_exact), 10.0f);
    printf("(EXACT)MSE for parameter: %f and bias: %f is : %f\n\n", w_exact, b_exact, cost(w_exact, b_exact));

    printf("MSE for randomised parameter: %f and bias: %f is : %f\n", w, b, cost(w, b));

    // janky fucking gradient descent
    for (size_t i = 0; i < ITERATIONS; i++) {
        // approx derivative:
        // https://wikimedia.org/api/rest_v1/media/math/render/svg/aae79a56cdcbc44af1612a50f06169b07f02cbf3
        // Using method of finite differences to find approximate derivatives of cost w.r.t w and b

        float dw = (cost(w + EPS, b) - cost(w, b)) / EPS; // ~ ∂C(w, b) / ∂w
        float db = (cost(w, EPS + b) - cost(w, b)) / EPS; // ~ ∂C(w, b) / ∂b

        // These approximations will approach actual derivatives as EPS -> 0
        // We move opposite to gradient to find a local minimum
        w -= LR_RATE * dw;
        b -= LR_RATE * db;
    }

    printf("Predicted value of y(for x = %f) = %f v/s real value: %f\n", 5.0f, MODEL(5, w, b), 10.0f);
    printf("MSE for parameter(after grad. descent) %f and bias %f is : %f\n", w, b, cost(w, b));

    printf("------------------------------------------\nEXACT RESULT:\nw = %f\tb = %f\n\
GRADIENT DESCENT RESULT:\nw = %f\tb = %f\n", w_exact, b_exact, w, b);

    return 0;
}