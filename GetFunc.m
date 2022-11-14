function val = GetFunc(Problem, A, b, lambda, x)

    if Problem == "LASSO"
        val = 0.5*norm(A*x-b)^2+lambda*norm(x,1);
    elseif Problem == "LASSO_SPARSE"
        val = 0.5*norm(A*x-b)^2+lambda*norm(x,1);
    elseif Problem == "Logistic"
        val = (-b'*A*x)+ones(1,size(b,1))*log(1+exp(A*x));
    elseif Problem == "Logistic_SPARSE"
        val = (-b'*A*x)+ones(1,size(b,1))*log(1+exp(A*x));
    elseif Problem == "NNLS"
        val = 0.5*norm(A*x-b)^2;
    elseif Problem == "Logistic l1"
        val = (-b'*A*x)+ones(1,size(b,1))*log(1+exp(A*x))+lambda*norm(x,1);
    elseif Problem == "Logistic_SPARSE l1"
        val = (-b'*A*x)+ones(1,size(b,1))*log(1+exp(A*x))+lambda*norm(x,1);
    elseif Problem == "Logsumexp"
        val = lambda*log(sum(exp((A*x-b)/lambda),"all"));
    elseif Problem == "Matrix Completion"
        val = 0.5*norm(A-x.*b,"fro")^2+lambda*sum(svd(x));
    elseif Problem == "Worst"
        val = (lambda-0)/8*(x(1,1)^2+x(size(x,1)/3,1)^2+norm(A*x)*norm(A*x)-2*x(1,1))+0/2*norm(x)^2;
    else
        val = 0;
    end
end