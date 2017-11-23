//
//  main.cpp
//  euler7
//
//  Created by Muhammad Rahim on 26/12/2016.
//  Copyright © 2016 Muhammad Rahim. All rights reserved.
//
/*
 By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13.
 
 What is the 10 001st prime number?
*/

#include <iostream>
#include <vector>

using namespace std;

int isprime(int test) {
    int divisor = 2;
    while (divisor < test/2) {
        if (test%divisor==0) {
            return false;
        }
        else {
            divisor++;
        }
        divisor = 2;
    }
    return true;
}

int storage() {
    int test = 7;
    vector<int> primes[] = 10001;
    while (1 < 2) {
        if (isprime(test) == true) {
        primes.push_back(test);
        }
        test++;
    }
    cout << primes[10001] << endl;
    return 0;
}


int main() {
    cout << storage() << endl;
    return 0;
}
