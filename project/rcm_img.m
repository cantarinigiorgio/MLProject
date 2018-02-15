function [out] = rcm_img(inputImg)
    %This function take as input an array that represent an RGB image 
    %and return the recomposed 3-channel image
    
    img_R = inputImg(1:1024);
    R = reshape(img_R,[32,32]);
    img_G = inputImg(1025:2048);
    G = reshape(img_G,[32,32]);
    img_B = inputImg(2049:3072);
    B = reshape(img_B,[32,32]);
    out = cat(3,R,G,B);
    %out = imrotate(out,90);
end

