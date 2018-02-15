function [hogvector1,hogvectorval1,hogvector2,hogvectorval2,hogvector3,hogvectorval3,hogvector4,hogvectorval4] = extractHOG(cellSize,X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val)
    %Extract HOG of all elements contained in each row of each input matrix
    
    if (cellSize == [4 4])
        m = 1764;
    end
    if (cellSize == [8 8])
        m = 324;
    end
    if (cellSize == [16 16])
        m = 36;
    end
    
    [~,sz2] = size(X1train);
    
    if(sz2 == 1024)%GRAYSCALE
        cellDim = [32,32];
    end
    if(sz2 == 3072)%RGB
        cellDim = [32,32,3];
    end
    
    [sz1,~] = size(X1train);
    nt = sz1;
    [sz1,~] = size(X1val);
    nv = sz1;
    
    hogvector1 = zeros (nt,m);
    for i = 1:nt
        hogvector1(i,:) = extractHOGFeatures(reshape(X1train(i,:),cellDim),'CellSize',cellSize);
    end

    hogvectorval1 = zeros (nv,m);
    for i = 1:nv
        hogvectorval1(i,:) = extractHOGFeatures(reshape(X1val(i,:),cellDim),'CellSize',cellSize);
    end

    hogvector2 = zeros (nt,m);
    for i = 1:nt
        hogvector2(i,:) = extractHOGFeatures(reshape(X2train(i,:),cellDim),'CellSize',cellSize);
    end
    hogvectorval2 = zeros (nv,m);
    for i = 1:nv
        hogvectorval2(i,:) = extractHOGFeatures(reshape(X2val(i,:),cellDim),'CellSize',cellSize);
    end

    hogvector3 = zeros (nt,m);
    for i = 1:nt
        hogvector3(i,:) = extractHOGFeatures(reshape(X3train(i,:),cellDim),'CellSize',cellSize);
    end
    
    hogvectorval3 = zeros (nv,m);
    for i = 1:nv
        hogvectorval3(i,:) = extractHOGFeatures(reshape(X3val(i,:),cellDim),'CellSize',cellSize);
    end
    
    hogvector4 = zeros (nt,m);
    for i = 1:nt
        hogvector4(i,:) = extractHOGFeatures(reshape(X4train(i,:),cellDim),'CellSize',cellSize);
    end
    
    hogvectorval4 = zeros (nv,m);
    for i = 1:nv
        hogvectorval4(i,:) = extractHOGFeatures(reshape(X4val(i,:),cellDim),'CellSize',cellSize);
    end

end

