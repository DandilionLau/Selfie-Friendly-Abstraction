function [vin, vout] = EdgeExtract(img, imgf)
    S = im2double(img);
    vin_v = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
    vin_h = [diff(S,1,2), S(:,1,:) - S(:,end,:)];
    vin = cat(3, vin_v, vin_h);
    
    img_filtered = im2double(imgf);
    S = im2double(img_filtered);
    
    vout_v = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
    vout_h = [diff(S,1,2), S(:,1,:) - S(:,end,:)];
    vout = cat(3, vout_v, vout_h);
end


