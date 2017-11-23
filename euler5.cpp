//
//  main.cpp
//  euler5
//
//  Created by Muhammad Rahim on 26/12/2016.
//  Copyright © 2016 Muhammad Rahim. All rights reserved.
//  Euler Problem 5:
/*
2520 is the smallest number that can be divided by each of the numbers from 1 to 10 without any remainder.
 
What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?
*/

#include <iostream>
#include <ctime>

using namespace std;

int division(int number) {
    int i = 2;
    clock_t begin = clock();
    while (i < 20) {
        if (number%i==0) {
            i++;
        }
        else {
            number++;
            i = 1;
        }
    }
    clock_t end = clock();
    cout << "Time taken " << double(end - begin)/CLOCKS_PER_SEC << endl;
    return number;
}

int main() {
    int number = 2;
    cout << division(number) << endl;
    return 0;
}
