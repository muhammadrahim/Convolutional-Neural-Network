//
//  main.cpp
//  euler4
//
//  Created by Muhammad Rahim on 08/12/2016.
//  Copyright © 2016 Muhammad Rahim. All rights reserved.
/*

 A palindromic number reads the same both ways. The largest palindrome made from the product of two 2-digit numbers is 9009 = 91  99.

Find the largest palindrome made from the product of two 3-digit numbers.
 
*/


#include <iostream>
using namespace std;


int reverse(int n) {
    int num = n;
    int rev = 0;
    do
    {
        rev = (rev * 10) + (num % 10);
        num = num / 10;
    } while (num > 0);
        return rev;
}

int palindrome(int n) {
    if (n == reverse(n)) {
        cout << n << endl;
        return 1;
    }
    return 0;
}

int main() {
    int largest = 0;
    int a = 100;
    while (a <= 999) {
        int b = 100;
            while (b <= 999) {
                if (palindrome(a*b) && a*b > largest) {
                largest = a*b;
                }
                b=b+1;
            }
        a=a+1;
    }
    cout << largest << endl;
    return 0;
}
