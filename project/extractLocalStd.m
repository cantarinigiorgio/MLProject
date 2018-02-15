function [lstdvector1,lstdvectorval1,lstdvector2,lstdvectorval2,lstdvector3,lstdvectorval3,lstdvector4,lstdvectorval4] = extractLocalStd(X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val)
    %Extract Local Standard Deviation of all elements contained in each row of each input matrix

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
    
    lstdvector1 = zeros (nt,m);

    for i = 1:nt
        lstdvector1(i,:) = reshape(stdfilt(reshape(X1train(i,:),cellDim)),[],m);%1024 cambiare con n per rgb
    end
    lstdvectorval1 = zeros (nv,m);
    for i = 1:nv
        lstdvectorval1(i,:) = reshape(stdfilt(reshape(X1val(i,:),cellDim)),[],m);
    end

    lstdvector2 = zeros (nt,m);
    for i = 1:nt
        lstdvector2(i,:) = reshape(stdfilt(reshape(X2train(i,:),cellDim)),[],m);
    end
    lstdvectorval2 = zeros (nv,m);
    for i = 1:nv
        lstdvectorval2(i,:) = reshape(stdfilt(reshape(X2val(i,:),cellDim)),[],m);
    end

    lstdvector3 = zeros (nt,m);
    for i = 1:nt
        lstdvector3(i,:) = reshape(stdfilt(reshape(X3train(i,:),cellDim)),[],m);
    end
    
    lstdvectorval3 = zeros (nv,m);
    for i = 1:nv
        lstdvectorval3(i,:) = reshape(stdfilt(reshape(X3val(i,:),cellDim)),[],m);
    end

    lstdvector4 = zeros (nt,m);
    for i = 1:nt
        lstdvector4(i,:) = reshape(stdfilt(reshape(X4train(i,:),cellDim)),[],m);
    end
    
    lstdvectorval4 = zeros (nv,m);
    for i = 1:nv
        lstdvectorval4(i,:) = reshape(stdfilt(reshape(X4val(i,:),cellDim)),[],m);
    end

end

