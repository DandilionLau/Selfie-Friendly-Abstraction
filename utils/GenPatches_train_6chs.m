addpath data/

clear;
patch_dim = 64;
num_patches = 10000;
listing = dir('data/train/selfie/*.jpg');

% load data 
for m = 1 : 40
    fprintf('Extracting patch batch: %d / %d\n', m, 50);
    % extract random patches
    samples = zeros(patch_dim, patch_dim, 6, num_patches);
    labels = zeros(size(samples));

    for i = 1 : num_patches / 8

        if (mod(i,100) == 0)
            fprintf('Extracting patch: %d / %d\n', i*8, num_patches);
        end
        
        r_idx = random('unid', size(listing, 1));
        I = imread(strcat('data/selfie/', listing(r_idx).name));
        I = imresize(I, [512 512]);
        Iout = imread(strrep(strcat('data/train/composition/', listing(r_idx).name), '.jpg', '.jpg'));        
        Iout = imresize(Iout, [512 512]);
        
        orig_img_size = size(I);
        r = random('unid', orig_img_size(1) - patch_dim + 1);
        c = random('unid', orig_img_size(2) - patch_dim + 1);
        
        patch = I(r:r+patch_dim-1, c:c+patch_dim-1, :);
        patchHoriFlipped = flipdim(patch, 2);
        
        patchOut = Iout(r:r+patch_dim-1, c:c+patch_dim-1, :);
        patcOuthHoriFlipped = flipdim(patchOut, 2);
        
        idx_list = (i-1)*8+1:(i-1)*8+8;
        for idx = 1:4
            patch_rotated = im2double(imrotate(patch, (idx-1)*90));
            patch_filtered = im2double(imrotate(patchOut, (idx-1)*90));
            [vin, vout] = EdgeExtract_6chs(im2double(patch_rotated), im2double(patch_filtered));
            samples(:,:,:,idx_list(idx)) = vin;
            labels(:,:,:,idx_list(idx)) = vout;            
            
            patch_rotated = im2double(imrotate(patchHoriFlipped, (idx-1)*90));
            patch_filtered = im2double(imrotate(patcOuthHoriFlipped, (idx-1)*90));
            [vin, vout] = EdgeExtract_6chs(im2double(patch_rotated), im2double(patch_filtered));            
            samples(:,:,:,idx_list(idx+4)) = vin;
            labels(:,:,:,idx_list(idx+4)) = vout;            
        end
    end
    samples = single(samples);
    labels = single(labels); 
    
    % save patches for training
    filename = strcat('data/composition/train_64_6chs/patches_', num2str(m));
    save(filename, '-v7.3', 'samples', 'labels');
end

