function [edgevector1,edgevectorval1,edgevector2,edgevectorval2,edgevector3,edgevectorval3,edgevector4,edgevectorval4] = extractEdges(X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val)
    %Extract edges of all elements contained in each row of each input matrix (if images are RGB or LAB convert them to grayscale)
    
    [~,sz2] = size(X1train);
    
    if (sz2 == 3072)
        X1train = rgbtog(X1train);
        X1val = rgbtog(X1val);
        X2train = rgbtog(X2train);
        X2val = rgbtog(X2val);
        X3train = rgbtog(X3train);
        X3val = rgbtog(X3val);
        X4train = rgbtog(X4train);
        X4val = rgbtog(X4val);
        X1val = double(X1val);  
        X2val = double(X2val); 
        X3val = double(X3val);   
        X4val = double(X4val);
        X1train = double(X1train);  
        X2train = double(X2train); 
        X3train = double(X3train);   
        X4train = double(X4train);
    end
    
    
    n=1024;% size sz2 di Xtrain
    
    [sz1,sz2] = size(X1train);
    nt = sz1;
    [sz1,sz2] = size(X1val);
    nv = sz1;
    
    
    edgevector1 = zeros(nt,n);
    
    for i = 1:nt
        edgevector1(i,:) = reshape(edge((reshape(X1train(i,:,:),[32,32]))),[],n);
    end
%     
    edgevectorval1 = zeros(nv,n);
    for i = 1:nv
        edgevectorval1(i,:) = reshape(edge((reshape(X1val(i,:,:),[32,32]))),[],n);
    end
    
    edgevector2 = zeros(nt,n);
    for i = 1:nt
        edgevector2(i,:) = reshape(edge((reshape(X2train(i,:,:),[32,32]))),[],n);
    end
%     
    edgevectorval2 = zeros(nv,n);
    for i = 1:nv
        edgevectorval2(i,:) = reshape(edge((reshape(X2val(i,:,:),[32,32]))),[],n);
    end
    
    edgevector3 = zeros(nt,n);
    for i = 1:nt
        edgevector3(i,:) = reshape(edge((reshape(X3train(i,:,:),[32,32]))),[],n);
    end
%     
    edgevectorval3 = zeros(nv,n);
    for i = 1:nv
        edgevectorval3(i,:) = reshape(edge((reshape(X3val(i,:,:),[32,32]))),[],n);
    end
    
     edgevector4 = zeros(nt,n);
    for i = 1:nt
        edgevector4(i,:) = reshape(edge((reshape(X4train(i,:,:),[32,32]))),[],n);
    end
%     
    edgevectorval4 = zeros(nv,n);
    for i = 1:nv
        edgevectorval4(i,:) = reshape(edge((reshape(X4val(i,:,:),[32,32]))),[],n);
    end
    
end

