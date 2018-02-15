function [featureExtracted,error] = extractFeatures_applySVMLAB(X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val)
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

     
    %%%%LOCAL RANGE %%agggiungere quarto valore
    [lrngvector1,lrngvectorval1,lrngvector2,lrngvectorval2,lrngvector3,lrngvectorval3,lrngvector4,lrngvectorval4] = extractLocalRange(X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val);
    err1 = applySVM(lrngvector1,lrngvectorval1);
    err2 = applySVM(lrngvector2,lrngvectorval2);
    err3 = applySVM(lrngvector3,lrngvectorval3);
    err4 = applySVM(lrngvector4,lrngvectorval4);

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
    

	%%%COMBINE localrange HOG4  
    [combinedlocalrangeHOG4v1,combinedlocalrangeHOG4vv1] = combineFeatures(lrngvector1,hog4v1,lrngvectorval1,hog4vv1);
    [combinedlocalrangeHOG4v2,combinedlocalrangeHOG4vv2] = combineFeatures(lrngvector2,hog4v2,lrngvectorval2,hog4vv2);
    [combinedlocalrangeHOG4v3,combinedlocalrangeHOG4vv3] = combineFeatures(lrngvector3,hog4v3,lrngvectorval3,hog4vv3);
    [combinedlocalrangeHOG4v4,combinedlocalrangeHOG4vv4] = combineFeatures(lrngvector4,hog4v4,lrngvectorval4,hog4vv4);

    err1 = applySVM(combinedlocalrangeHOG4v1,combinedlocalrangeHOG4vv1);
    err2 = applySVM(combinedlocalrangeHOG4v2,combinedlocalrangeHOG4vv2);
    err3 = applySVM(combinedlocalrangeHOG4v3,combinedlocalrangeHOG4vv3);
    err4 = applySVM(combinedlocalrangeHOG4v4,combinedlocalrangeHOG4vv4);
    
    meanerrrangeHOG4 = (err1+err2+err3+err4)/4;
    meanErr(i).ft = "RNGH4";
    meanErr(i).err = meanerrrangeHOG4;

    i=i+1;
    
	%%%%localrange HOG8  
    [combinedlocalrangeHOG8v1,combinedlocalrangeHOG8vv1] = combineFeatures(lrngvector1,hog8v1,lrngvectorval1,hog8vv1);
    [combinedlocalrangeHOG8v2,combinedlocalrangeHOG8vv2] = combineFeatures(lrngvector2,hog8v2,lrngvectorval2,hog8vv2);
    [combinedlocalrangeHOG8v3,combinedlocalrangeHOG8vv3] = combineFeatures(lrngvector3,hog8v3,lrngvectorval3,hog8vv3);
    [combinedlocalrangeHOG8v4,combinedlocalrangeHOG8vv4] = combineFeatures(lrngvector4,hog8v4,lrngvectorval4,hog8vv4);

    
    err1 = applySVM(combinedlocalrangeHOG8v1,combinedlocalrangeHOG8vv1);
    err2 = applySVM(combinedlocalrangeHOG8v2,combinedlocalrangeHOG8vv2);
    err3 = applySVM(combinedlocalrangeHOG8v3,combinedlocalrangeHOG8vv3);
    err4 = applySVM(combinedlocalrangeHOG8v4,combinedlocalrangeHOG8vv4);

    meanerrrangeHOG8 = (err1+err2+err3+err4)/4;
    
    meanErr(i).ft = "RNGH8";
    meanErr(i).err = meanerrrangeHOG8;
    
    i=i+1;     
    
    
    %%%%COMBINE localstd HOG4  
    [combinedlocalstdHOG4v1,combinedlocalstdHOG4vv1] = combineFeatures(lstdvector1,hog4v1,lstdvectorval1,hog4vv1);
    [combinedlocalstdHOG4v2,combinedlocalstdHOG4vv2] = combineFeatures(lstdvector2,hog4v2,lstdvectorval2,hog4vv2);
    [combinedlocalstdHOG4v3,combinedlocalstdHOG4vv3] = combineFeatures(lstdvector3,hog4v3,lstdvectorval3,hog4vv3);
    [combinedlocalstdHOG4v4,combinedlocalstdHOG4vv4] = combineFeatures(lstdvector4,hog4v4,lstdvectorval4,hog4vv4);
    
    err1 = applySVM(combinedlocalstdHOG4v1,combinedlocalstdHOG4vv1);
    err2 = applySVM(combinedlocalstdHOG4v2,combinedlocalstdHOG4vv2);
    err3 = applySVM(combinedlocalstdHOG4v3,combinedlocalstdHOG4vv3);
    err4 = applySVM(combinedlocalstdHOG4v4,combinedlocalstdHOG4vv4);

    meanerrstdHOG4 = (err1+err2+err3+err4)/4;
    meanErr(i).ft = "STDH4";
    meanErr(i).err = meanerrstdHOG4;

    i=i+1;
    
    
    %%%COMBINE localstd HOG8  
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
    

    auxft = [];
    auxerr = [];
    
    for i = 1:10
        auxft = cat(1,auxft,meanErr(i).ft);
        auxerr = cat(1,auxerr,meanErr(i).err);
    end
    
    [error,I] = sort(auxerr);%,'descend');%error sorted
    featureExtracted = [];
    
    for i = 1:10
        featureExtracted = cat(1,featureExtracted,auxft(I(i,1)));%features sorted
    end

end

