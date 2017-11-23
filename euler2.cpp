//
//  main.cpp
//  Even Fibonacci numbers
//
//  Created by Muhammad Rahim on 26/11/2016.
//  Copyright © 2016 Muhammad Rahim. All rights reserved.
//

#include <iostream>

using namespace std;

int main() {
    int sum = 0;
    int i = 1;
    int j = 1;
    int k;
    
    while(j <= 4000000) {
        
        if(j % 2 == 0)
            sum += j;
        
        k=j;
        j += i;
        i=k;
        cout << sum << endl;
    }
    cout << sum << endl;
    return 0;
}

