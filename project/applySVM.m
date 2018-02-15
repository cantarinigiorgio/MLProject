function [err] = applySVM(X,Xval)
    %Apply svm to X and the calculate the loss with Xval
    
    [sz,~] = size(X);
    Y = cat(1,ones(sz/2,1),zeros(sz/2,1));

    SVMModel = fitcsvm(X,Y);
    [sz1,~] = size(Xval);
    Yval = cat(1,ones(sz1/2,1),zeros(sz1/2,1));
    err = loss(SVMModel,Xval,Yval);

end

