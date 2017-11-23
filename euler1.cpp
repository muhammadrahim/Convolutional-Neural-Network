//
//  main.cpp
//  euler1
//
//  Created by Muhammad Rahim on 26/11/2016.
//  Copyright © 2016 Muhammad Rahim. All rights reserved.
//

#include <iostream>

using namespace std;

int main() {
    unsigned int i;
    int sum = 0;
    for (i=0;i<1000;i++) {
        if (i%3==0 || i%5==0)
            sum=sum+i;
    }
    cout << sum << endl;
    return 0;
}
