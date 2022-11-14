function grad = GetGrad(Problem, A, b, lambda, x, t)

    if Problem == "LASSO"
        grad = (x-wthresh(t*A'*(b-A*x)+x,'s',t*lambda))/t;
    elseif Problem == "LASSO_SPARSE"
        y = A'*(b-A*x);
        grad = (x-wthresh(t*y+x,'s',t*lambda))/t;
    elseif Problem == "Logistic"
        grad = A'*(-b+sigmoid(A*x));
    elseif Problem == "Logistic_SPARSE"
        grad = A'*(-b+sigmoid(A*x));
    elseif Problem == "NNLS"
        grad = (x-max(0,x-t*(A'*A*x-A'*b)))/t;
    elseif Problem == "Logistic l1"
        grad = (x-wthresh(t*A'*(b-sigmoid(A*x))+x,'s',t*lambda))/t;
    elseif Problem == "Logistic_SPARSE l1"
        grad = (x-wthresh(t*A'*(b-sigmoid(A*x))+x,'s',t*lambda))/t;
    elseif Problem == "Logsumexp"
        grad = A'*exp((A*x-b)/lambda)/sum(exp((A*x-b)/lambda),"all");
    elseif Problem == "Matrix Completion"
        [U,S,V] = svd(x+t*(A-x.*b));
        Sl = max(S-t*lambda,0);
        grad = (x-U*Sl*V')/t;
    elseif Problem == "Worst"
        u = zeros(size(x,1),1);
        u(1,1) = 1;
        v = zeros(size(x,1),1);
        v(size(x,1)/3,1) = 1;
        grad = (lambda-0)/4*(A'*A)*x+(lambda-0)*(x(1,1)-1)*u/4+lambda*x(size(x,1)/3,1)*v/4+0*x;
    else
        grad = zeros(size(x));
    end
end