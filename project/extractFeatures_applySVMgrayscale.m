function [featureExtracted,error] = extractFeatures_applySVMgrayscale(X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val)
    %For each fold received, extract features and then apply SVM algorithm for classification
    %At least take the mean of the four errors
    
    %initialize variable that will be used for the returned value of the
    %function
    
    s.ft = "";
    s.err = 0;
    meanErr = repmat(s,1,10);
    i = 1;

    
    %%%EDGES
    
    
    [edgev1,edgevv1,edgev2,edgevv2,edgev3,edgevv3,edgev4,edgevv4] = extractEdges(X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val);
    
    err1 = applySVM(edgev1,edgevv1);
    err2 = applySVM(edgev2,edgevv2);
    err3 = applySVM(edgev3,edgevv3);
    err4 = applySVM(edgev4,edgevv4);
    meanerrEdges = (err1+err2+err3+err4)/4;
   
    meanErr(i).ft = "E";
    meanErr(i).err = meanerrEdges;

    i=i+1;

    
    %%%%%%HOG     [4 4] 1x1764 ; [8 8] 1x324 ; [16 16] 1x36
    
    
    %%%HOG 4
    cellSize = [4 4];
    [hog4v1,hog4vv1,hog4v2,hog4vv2,hog4v3,hog4vv3,hog4v4,hog4vv4] = extractHOG(cellSize,X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val);

    err1 = applySVM(hog4v1,hog4vv1);
    err2 = applySVM(hog4v2,hog4vv2);
    err3 = applySVM(hog4v3,hog4vv3);
    err4 = applySVM(hog4v4,hog4vv4);
    
    meanerrHOG4 = (err1+err2+err3+err4)/4;
    
    meanErr(i).ft = "H4";
    meanErr(i).err = meanerrHOG4;
    i=i+1;

    
    %%%HOG 8
    cellSize = [8 8];
    [hog8v1,hog8vv1,hog8v2,hog8vv2,hog8v3,hog8vv3,hog8v4,hog8vv4] = extractHOG(cellSize,X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val);

    err1 = applySVM(hog8v1,hog8vv1);
    err2 = applySVM(hog8v2,hog8vv2);
    err3 = applySVM(hog8v3,hog8vv3);
    err4 = applySVM(hog8v4,hog8vv4);
    
    meanerrHOG8 = (err1+err2+err3+err4)/4;
    
    meanErr(i).ft = "H8";
    meanErr(i).err = meanerrHOG8;

    i=i+1;

    %%%HOG 16
    cellSize = [16 16];
    [hog16v1,hog16vv1,hog16v2,hog16vv2,hog16v3,hog16vv3,hog16v4,hog16vv4] = extractHOG(cellSize,X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val);

    err1 = applySVM(hog16v1,hog16vv1);
    err2 = applySVM(hog16v2,hog16vv2);
    err3 = applySVM(hog16v3,hog16vv3);
    err4 = applySVM(hog16v4,hog16vv4);

    meanerrHOG16 = (err1+err2+err3+err4)/4;
    
    meanErr(i).ft = "H16";
    meanErr(i).err = meanerrHOG16;

    i=i+1;

    
    %%%%LOCAL RANGE 
    [lrngvector1,lnrgvectorval1,lrngvector2,lnrgvectorval2,lnrgvector3,lnrgvectorval3,lnrgvector4,lnrgvectorval4] = extractLocalRange(X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val);
    err1 = applySVM(lrngvector1,lnrgvectorval1);
    err2 = applySVM(lrngvector2,lnrgvectorval2);
    err3 = applySVM(lnrgvector3,lnrgvectorval3);
    err4 = applySVM(lnrgvector4,lnrgvectorval4);

    meanerrLocalRange = (err1+err2+err3+err4)/4;
    
    meanErr(i).ft = "RNG";
    meanErr(i).err = meanerrLocalRange;

    i=i+1;
 

    %%%%LOCAL STD
    [lstdvector1,lstdvectorval1,lstdvector2,lstdvectorval2,lstdvector3,lstdvectorval3,lstdvector4,lstdvectorval4] = extractLocalStd(X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val);
    err1 = applySVM(lstdvector1,lstdvectorval1);
    err2 = applySVM(lstdvector2,lstdvectorval2);
    err3 = applySVM(lstdvector3,lstdvectorval3);
    err4 = applySVM(lstdvector4,lstdvectorval4);

    meanerrLocalSTD = (err1+err2+err3+err4)/4;
    
    meanErr(i).ft = "STD";
    meanErr(i).err = meanerrLocalSTD;

    i=i+1;
    
    
    %%%%COMBINE localstd HOG8  
    [combinedlocalstdHOG8v1,combinedlocalstdHOG8vv1] = combineFeatures(lstdvector1,hog8v1,lstdvectorval1,hog8vv1);
    [combinedlocalstdHOG8v2,combinedlocalstdHOG8vv2] = combineFeatures(lstdvector2,hog8v2,lstdvectorval2,hog8vv2);
    [combinedlocalstdHOG8v3,combinedlocalstdHOG8vv3] = combineFeatures(lstdvector3,hog8v3,lstdvectorval3,hog8vv3);
    [combinedlocalstdHOG8v4,combinedlocalstdHOG8vv4] = combineFeatures(lstdvector4,hog8v4,lstdvectorval4,hog8vv4);
    
    err1 = applySVM(combinedlocalstdHOG8v1,combinedlocalstdHOG8vv1);
    err2 = applySVM(combinedlocalstdHOG8v2,combinedlocalstdHOG8vv2);
    err3 = applySVM(combinedlocalstdHOG8v3,combinedlocalstdHOG8vv3);
    err4 = applySVM(combinedlocalstdHOG8v4,combinedlocalstdHOG8vv4);
    
    meanerrstdHOG8 = (err1+err2+err3+err4)/4;
    
    meanErr(i).ft = "STDH8";
    meanErr(i).err = meanerrstdHOG8;
    
    i=i+1;
    
   
	%%%LBP    [8 8] 1x944

    cellSize = [8 8];
    [lbp8v1,lbp8vv1,lbp8v2,lbp8vv2,lbp8v3,lbp8vv3,lbp8v4,lbp8vv4] = extractLBP(cellSize,X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val);
   
    err1 = applySVM(lbp8v1,lbp8vv1);
    err2 = applySVM(lbp8v2,lbp8vv2);
    err3 = applySVM(lbp8v3,lbp8vv3);
    err4 = applySVM(lbp8v4,lbp8vv4);

    meanerrLBP8 = (err1+err2+err3+err4)/4;
    
    meanErr(i).ft = "LBP8";
    meanErr(i).err = meanerrLBP8;
    i=i+1;

    %%%%COMBINE edges HOG4  
    [combinededgesHOG4v1,combinededgesHOG4vv1] = combineFeatures(edgev1,hog4v1,edgevv1,hog4vv1);
    [combinededgesHOG4v2,combinededgesHOG4vv2] = combineFeatures(edgev2,hog4v2,edgevv2,hog4vv2);
    [combinededgesHOG4v3,combinededgesHOG4vv3] = combineFeatures(edgev3,hog4v3,edgevv3,hog4vv3);
    [combinededgesHOG4v4,combinededgesHOG4vv4] = combineFeatures(edgev4,hog4v4,edgevv4,hog4vv4);

    err1 = applySVM(combinededgesHOG4v1,combinededgesHOG4vv1);
    err2 = applySVM(combinededgesHOG4v2,combinededgesHOG4vv2);
    err3 = applySVM(combinededgesHOG4v3,combinededgesHOG4vv3);
    err4 = applySVM(combinededgesHOG4v4,combinededgesHOG4vv4);

    meanerredgesHOG4 = (err1+err2+err3+err4)/4;
    
    meanErr(i).ft = "H4E";
    meanErr(i).err = meanerredgesHOG4;
    i=i+1;
        
    
    %%%%%COMBINE edges HOG8  
    [combinededgesHOG8v1,combinededgesHOG8vv1] = combineFeatures(edgev1,hog8v1,edgevv1,hog8vv1);
    [combinededgesHOG8v2,combinededgesHOG8vv2] = combineFeatures(edgev2,hog8v2,edgevv2,hog8vv2);
    [combinededgesHOG8v3,combinededgesHOG8vv3] = combineFeatures(edgev3,hog8v3,edgevv3,hog8v3);
    [combinededgesHOG8v4,combinededgesHOG8vv4] = combineFeatures(edgev4,hog8v4,edgevv4,hog8vv4);
    
    err1 = applySVM(combinededgesHOG8v1,combinededgesHOG8vv1);
    err2 = applySVM(combinededgesHOG8v2,combinededgesHOG8vv2);
    err3 = applySVM(combinededgesHOG8v3,combinededgesHOG8vv3);
    err4 = applySVM(combinededgesHOG8v4,combinededgesHOG8vv4);

    meanerredgesHOG8 = (err1+err2+err3+err4)/4;
    
    meanErr(i).ft = "H8E";
    meanErr(i).err = meanerredgesHOG8;

    
    auxft = [];
    auxerr = [];
    
    for i = 1:10
        meanErr(i).err
        auxft = cat(1,auxft,meanErr(i).ft);
        auxerr = cat(1,auxerr,meanErr(i).err);
    end
    
    [error,I] = sort(auxerr);%,'descend');%error ordered
    featureExtracted = [];
    
    for i = 1:10
        featureExtracted = cat(1,featureExtracted,auxft(I(i,1)));%features ordered
    end

end

