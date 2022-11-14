a = 1
us = 0.0001
ks = 1000

for i =1:ks

    a = a*(1-us-i*us*us);
    
end

a
b = 1/(1+ks*ks*us*us)
