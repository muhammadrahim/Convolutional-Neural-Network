//
//  main.cpp
//  euler3
//
//  Created by Muhammad Rahim on 27/11/2016.
//  Copyright © 2016 Muhammad Rahim. All rights reserved.
//

#include <iostream>

using namespace std;

#define COMP 600851475143

int main() {
    
    unsigned long long int i = 2;
    unsigned long long int max = 0;
    
    while(i*i <= COMP) {
        
        if(COMP % i == 0) {
            
            bool check = true;
            
            for(unsigned int n = 2; n*n <= i; ++n) {
                
                if(i % n == 0) {
                    check = false;
                    break;
                }
                
            }
            
            if(check)
                max = i;
            
        }
        ++i;
        
    }
    
    if(COMP % max == 0)
        cout << max << endl;
    
    return 0;
    
}