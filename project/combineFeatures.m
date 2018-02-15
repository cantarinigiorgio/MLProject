function [combined1,combined2] = combineFeatures(edgev1,hogv1,edgevv1,hogvv1)
    %Combine features associating into a unique vector two different
    %vector(describing a specific feature)
    
    [~,s1] = size(edgev1);
    [~,s2] = size(hogv1);
    
    [sz1,~] = size(edgev1);
    nt = sz1;
    [sz1,~] = size(edgevv1);
    nv = sz1;
    
    sz = s1+s2;
    combined1=zeros(nt,sz);
     for i = 1:nt
        combined1(i,:) = cat(2,edgev1(i,:),hogv1(i,:));
     end
     
     combined2=zeros(nv,sz);
     for i = 1:nv
        combined2(i,:) = cat(2,edgevv1(i,:),hogvv1(i,:));
     end
end

