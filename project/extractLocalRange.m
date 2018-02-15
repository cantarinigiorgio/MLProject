function [rngvector1,lrngvectorval1,lrngvector2,lrngvectorval2,lrngvector3,lrngvectorval3,lrngvector4,lrngvectorval4] = extractHOG(X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val)
    %Extract Local Range of all elements contained in each row of each input matrix

    [~,sz2] = size(X1train);
    
    if(sz2 == 1024)%GRAYSCALE
        cellDim = [32,32];
        m = 1024;
    end
    
    if(sz2 == 3072)%RGB
        cellDim = [32,32,3];
        m = 3072;
    end
    
    [sz1,sz2] = size(X1train);
    nt = sz1;
    [sz1,sz2] = size(X1val);
    nv = sz1;

    rngvector1 = zeros (nt,m);
 
    for i = 1:nt
        rngvector1(i,:) = reshape(rangefilt(reshape(X1train(i,:),cellDim)),[],m);
    end

    lrngvectorval1 = zeros (nv,m);
    for i = 1:nv
        lrngvectorval1(i,:) = reshape(rangefilt(reshape(X1val(i,:),cellDim)),[],m);
    end

    lrngvector2 = zeros (nt,m);
    for i = 1:nt
        lrngvector2(i,:) = reshape(rangefilt(reshape(X2train(i,:),cellDim)),[],m);
    end
    lrngvectorval2 = zeros (nv,m);
    for i = 1:nv
        lrngvectorval2(i,:) = reshape(rangefilt(reshape(X2val(i,:),cellDim)),[],m);
    end

    lrngvector3 = zeros (nt,m);
    for i = 1:nt
        lrngvector3(i,:) = reshape(rangefilt(reshape(X3train(i,:),cellDim)),[],m);
    end
    
    lrngvectorval3 = zeros (nv,m);
    for i = 1:nv
        lrngvectorval3(i,:) = reshape(rangefilt(reshape(X3val(i,:),cellDim)),[],m);
    end

    lrngvector4 = zeros (nt,m);
    for i = 1:nt
        lrngvector4(i,:) = reshape(rangefilt(reshape(X4train(i,:),cellDim)),[],m);
    end
    
    lrngvectorval4 = zeros (nv,m);
    for i = 1:nv
        lrngvectorval4(i,:) = reshape(rangefilt(reshape(X4val(i,:),cellDim)),[],m);
    end

end

