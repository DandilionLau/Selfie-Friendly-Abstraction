addpath train/
addpath test/
addpath utils/
addpath results/

caffe.reset_all();
% CPU mode
% caffe.set_mode_cpu();

% GPU mode
caffe.set_mode_gpu();
caffe.set_device(0);

% dream
weights = 'model/style/composition.caffemodel';

mask = zeros(512,512,512,1);

% mask for subpixel layer, may take a bit time to initailize
for d1 = 1:1
    for d2 = 1:512
        for d3 = 1:512
            for d4 = 1:512
                mask(d4,d3,d2,d1) = 0;
                if(mod(d4,4) + 1 == ceil(ceil(d2/32)/4))
                    if(mod(d3,4) == mod(ceil(d2/32),4))
                        mask(d4,d3,d2,d1) = 1;
                    end
                end
            end
        end
    end
    
end

model = 'test/net_reshape.prototxt';

net = caffe.Net(model, 'test');

% load every image for the testing step
for m = 1:99
I = im2double(imread(strcat('data/test/', num2str(m), '.png')));

% resize the input to fit the mask
I = imresize(I, [512 512]);

% beta for the reconstruction step
beta = 10;

S = I;

tic
v_input = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
h_input = [diff(S,1,2), S(:,1,:) - S(:,end,:)];

v_input = permute(v_input, [2 1 3 4]);
v_input = v_input(:, :, [3, 2, 1], :);
h_input = permute(h_input, [2 1 3 4]);
h_input = h_input(:, :, [3, 2, 1], :);

input = cat(3, v_input, h_input);

net.blobs('samples').set_data(input);
net.blobs('mask').set_data(mask);

filtered = zeros(size(I,1), size(I,2), size(I,3), 2);

    for n = 1:1

    % loading the weights
    net.copy_from(weights);

    net.forward_prefilled();
    output = net.blobs('conv9').get_data();
    v = output(:,:,1:3);
    h = output(:,:,4:6);

    v = permute(v, [2 1 3 4]);
    h = permute(h, [2 1 3 4]);
    v = v(:, :, [3, 2, 1], :);
    h = h(:, :, [3, 2, 1], :);


    h(:, end, :) = S(:,1,:) - S(:,end,:);
    v(end, :, :) = S(1,:,:) - S(end,:,:);
    toc

    tic
    % recostuction step
    filtered(:,:,:,n) = grad_process(S, v, h, beta);
    toc

    % write filtered image  
    imwrite(filtered(:,:,:,1), strcat('result/composition/', num2str(m),'.jpg'));

    end
    
end



