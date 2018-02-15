function [lbpvector1,lbpvectorval1,lbpvector2,lbpvectorval2,lbpvector3,lbpvectorval3,lbpvector4,lbpvectorval4] = extractLBP(cellSize,X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val)
    % Extract LBP of all elements contained in each row of each input matrix

%      [~,sz2] = size(X1train);

%      if (sz2 == 3072)
%         X1train = atorgb(X1train);
%         X1val = atorgb(X1val);
%         X2train = atorgb(X2train);
%         X2val = atorgb(X2val);
%         X3train = atorgb(X3train);
%         X3val = atorgb(X3val);
%     end
    
    if (cellSize == [32 32])
        m = 59;
    end
    if (cellSize == [16 16])
        m = 236;
    end
    if (cellSize == [8 8])
        m = 944;
    end
    
    cellDim = [32,32];
    
    [sz1,~] = size(X1train);
    nt = sz1;
    [sz1,~] = size(X1val);
    nv = sz1;
    
    lbpvector1 = zeros (nt,m);
    for i = 1:nt
        lbpvector1(i,:) = extractLBPFeatures(reshape(X1train(i,:),cellDim),'CellSize',cellSize,'Normalization','None');
    end
    
    lbpvectorval1 = zeros (nv,m);
    for i = 1:nv
        lbpvectorval1(i,:) = extractLBPFeatures(reshape(X1val(i,:),cellDim),'CellSize',cellSize,'Normalization','None');
    end

    lbpvector2 = zeros (nt,m);
    for i = 1:nt
        lbpvector2(i,:) = extractLBPFeatures(reshape(X2train(i,:),cellDim),'CellSize',cellSize,'Normalization','None');
    end
    lbpvectorval2 = zeros (nv,m);
    for i = 1:nv
        lbpvectorval2(i,:) = extractLBPFeatures(reshape(X2val(i,:),cellDim),'CellSize',cellSize,'Normalization','None');
    end

    lbpvector3 = zeros (nt,m);
    for i = 1:nt
        lbpvector3(i,:) = extractLBPFeatures(reshape(X3train(i,:),cellDim),'CellSize',cellSize,'Normalization','None');
    end
    lbpvectorval3 = zeros (nv,m);
    
    for i = 1:nv
        lbpvectorval3(i,:) = extractLBPFeatures(reshape(X3val(i,:),cellDim),'CellSize',cellSize,'Normalization','None');
    end
    
    lbpvector4 = zeros (nt,m);
    for i = 1:nt
        lbpvector4(i,:) = extractLBPFeatures(reshape(X4train(i,:),cellDim),'CellSize',cellSize,'Normalization','None');
    end
    lbpvectorval4 = zeros (nv,m);
    
    for i = 1:nv
        lbpvectorval4(i,:) = extractLBPFeatures(reshape(X4val(i,:),cellDim),'CellSize',cellSize,'Normalization','None');
    end

end

