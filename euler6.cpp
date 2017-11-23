//
//  main.cpp
//  euler6
//
//  Created by Muhammad Rahim on 26/12/2016.
//  Copyright © 2016 Muhammad Rahim. All rights reserved.
//
/*
 The sum of the squares of the first ten natural numbers is,
 
 1^2 + 2^2 + ... + 10^2 = 385
 The square of the sum of the first ten natural numbers is,
 
 (1 + 2 + ... + 10)^2 = 55^2 = 3025
 Hence the difference between the sum of the squares of the first ten natural numbers and the square of the sum is 3025 − 385 = 2640.
 
 Find the difference between the sum of the squares of the first one hundred natural numbers and the square of the sum.
*/

#include <iostream>

using namespace std;

int squaresum() {
    int j = 0;
    for (int i=1;i<=100;i++) {
        j = j + i;
    }
    int squaresumm = j*j;
    return squaresumm;
}

int sum() {
    int summ = 0;
    int j = 0;
    for (int i=1;i<=100;i++) {
        j = i*i;
        summ += j;
    }
    return summ;
}

int main() {
    cout << (squaresum() - sum()) << endl;
    return 0;
}
