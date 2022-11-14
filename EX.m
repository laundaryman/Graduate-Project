N = 5000;
M = 50000;
A = sprand(N, M, 0.005)*0.2;
%A = rand(N,M);
x = randn(M,1);
z = randn(N,1);
b = A*x+z;
tic
y = A'*(b-A*x);
(x-wthresh(t*y+x,'s',t*lambda))/t;
toc


