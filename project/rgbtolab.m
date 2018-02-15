function [outputArg] = rgbtolab(inputArg)
    %Converts a matrix containing for each row an RGB image into a matrix
    %containing for each row a LAB image using function rgb2lab()
    
    labtrain = zeros(60000,32,32,3,'uint8');

    for i = 1:60000 
        aux = rcm_img(inputArg(i,:,:,:));
        labtrain(i,:,:,:) = rgb2lab(aux);
    end
    
    outputArg = labtrain;
    
end

