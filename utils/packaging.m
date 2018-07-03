addpath /home/jimmyren/caffe_std_new/caffe/matlab/
addpath data/dream/
addpath train/
addpath test/
addpath utils/

caffe.reset_all();
% CPU mode
%caffe.set_mode_cpu();

% GPU mode
caffe.set_mode_gpu();
caffe.set_device(0);

solver = caffe.Solver('train/solver_content_only.prototxt');

% dream
%weights = 'results/content_gan/dream/snapshot__iter_330000.caffemodel';
%save_weights = 'results/content_gan/dream/dr.caffemodel';

% composition
%weights = 'results/content_gan/composition/snapshot__iter_348000.caffemodel';
%save_weights = 'results/content_gan/composition/co.caffemodel';

% curly hair
%weights = 'results/content_gan/curly_hair/snapshot__iter_449000.caffemodel';
%save_weights = 'results/content_gan/curly_hair/cu.caffemodel';

% dallas
%weights = 'results/content_gan/dallas/snapshot__iter_107000.caffemodel';
%save_weights = 'results/content_gan/dallas/da.caffemodel';

% gothic
%weights = 'results/content_gan/Gothic/snapshot__iter_82000.caffemodel';
%save_weights = 'results/content_gan/Gothic/go.caffemodel';

% TransverseLine
%weights = 'results/content_gan/TransverseLine/snapshot__iter_10000.caffemodel';
%save_weights = 'results/content_gan/TransverseLine/tr.caffemodel';

% Femme
%weights = 'results/content_gan/Femme/snapshot__iter_71000.caffemodel';
%save_weights = 'results/content_gan/Femme/fe.caffemodel';

% Udnie
weights = 'results/content_gan/Udnie/snapshot__iter_80000.caffemodel';
save_weights = 'results/content_gan/Udnie/ud.caffemodel';


model = 'test/net_ss_gan_deploy.prototxt';
net = caffe.Net(model, weights, 'test');
net.save(save_weights);


