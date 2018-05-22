addpath data/dream/
addpath train/
addpath test/
addpath utils/

caffe.reset_all();
% CPU mode
% caffe.set_mode_cpu();

% GPU mode
caffe.set_mode_gpu();
caffe.set_device(0);

solver = caffe.Solver('train/solver_reshape.prototxt');

% if you want to train from initialization
existing_weights = 'model/vgg_16layers/vgg16.caffemodel';

% if you want to use existing weight
% existing_weights = 'model/style/composition.caffemodel';

solver.net.copy_from(existing_weights);

% load data
load('data/train/composition/val_64_6chs/patches_1.mat');
test_samples =samples(:,:,:,1:1000);
test_labels = labels(:,:,:,1:1000);

% prepare the mask
mask = zeros(64,64,512,10);

% initialize the mask
for d1 = 1:10
    for d2 = 1:512
        for d3 = 1:64
            for d4 = 1:64
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

% batch size
batch_size = 10;

% pass for training
for pass = 1:100
    fprintf('%d round data passing\n', pass);
    % Load patches
    for file = 1:49
        load(strcat('data/train/composition/train_64_6chs/patches_', num2str(file), '.mat'));
    
        % permute the data
        perm = randperm(size(samples,4));
        samples = samples(:,:,:,perm);
        labels = labels(:,:,:,perm);    

        for batch = 1:size(samples,4)/batch_size
            start_idx = (batch-1) * batch_size + 1;
            
            img_batch = samples(:,:,:,start_idx:start_idx+batch_size-1);        
            labels_ = labels(:,:,:,start_idx:start_idx+batch_size-1);                
            % permute dimension
            samples_ = permute(img_batch, [2 1 3 4]);
            samples_ = samples_(:, :, [3, 2, 1, 6, 5, 4], :);
            labels_ = permute(labels_, [2 1 3 4]);
            labels_ = labels_(:, :, [3, 2, 1, 6, 5, 4], :);

            solver.net.blobs('samples').set_data(samples_);
            solver.net.blobs('labels').set_data(labels_);
            solver.net.blobs('mask').set_data(mask);
            
            solver.step(1);
            iter = solver.iter();
            
	    % testing step
            if(mod(iter, 1000) == 0)  
                cost_feat = 0;
                cost_pixel = 0;
                fprintf('\ntesting batch: ');
                for tst_batch = 1:size(test_samples,4)/batch_size
                    fprintf('%d ', tst_batch);
                    start_idx = (tst_batch-1) * batch_size + 1;

                    img_batch = test_samples(:,:,:,start_idx:start_idx+batch_size-1);                
                    test_labels_ = test_labels(:,:,:,start_idx:start_idx+batch_size-1);
                
                    % permute dimension
                    test_samples_ = permute(img_batch, [2 1 3 4]);
                    test_samples_ = test_samples_(:, :, [3, 2, 1, 6, 5, 4], :);
                    test_labels_ = permute(test_labels_, [2 1 3 4]);
                    test_labels_ = test_labels_(:, :, [3, 2, 1, 6, 5, 4], :);                    
                    
                    solver.test_nets(1).blobs('samples').set_data(test_samples_);
                    solver.test_nets(1).blobs('labels').set_data(test_labels_);
                    solver.test_nets(1).blobs('mask').set_data(mask);

                    solver.test_nets(1).forward_prefilled();
                    cost_feat_tmp = solver.test_nets(1).blobs('loss_feat').get_data();
                    cost_feat = cost_feat + cost_feat_tmp;
                    cost_pixel_tmp = solver.test_nets(1).blobs('loss_pixel').get_data();
                    cost_pixel = cost_pixel + cost_pixel_tmp;
                    
                end
                cost_feat = cost_feat / (size(test_samples,4)/batch_size);    
                cost_pixel = cost_pixel / (size(test_samples,4)/batch_size);    
                fprintf('\ntesting acc: cost_feat: %f; cost_pixel: %f;', cost_feat, cost_pixel);
            end      
        end
    end
end



