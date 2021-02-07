clc; clear all;

%% dataset information
% datapath = './data/';
% MNIST_USPS_surf dataset, MNIST:2000*256, USPS:1800*256
srcStr = {'MNIST','USPS'};
tarStr = {'USPS', 'MNIST'};

%% set hyper-parameters
options.d = 20;        % #subspace dimension, default=20
options.p = 5;         % #neighbors, default=5
options.T = 10;        % #iterations, default=10
options.gamma = 0.01;  % gamma in paper, keep default  
options.eta = 10;      % eta in paper, keep default
options.rho = 1;       % rho in paper, keep default
options.mu = 0.1;      % mu in paper, keep default

accuracy_list = [];                                                    
for i = 1:2
    src = char(srcStr{i});
    tar = char(tarStr{i});
    
    % load source domian dataset
    load(['./data/MNIST_USPS/' src '.mat']);
    Xs = fts;  % n*m
    Ys = labels;  % n*1
    clear fts, clear labels;
    
    % load target domain dataset
    load(['./data/MNIST_USPS/' tar '.mat']);
    Xt = fts;  % n*m
    Yt = labels;  % n*1
    clear fts; clear labels;
    
    % data preprocessing    
    Xs = Xs';  % dim*n
    Xt = Xt';  % dim*n
    Xs = Xs*diag(sparse(1./sqrt(sum(Xs.^2))));  % normalization
    Xt = Xt*diag(sparse(1./sqrt(sum(Xt.^2))));
    Xs = Xs';  % n*dim
    Xt = Xt';  % n*dim

    %% MDTL
    [acc] = MDTL(Xs,Ys,Xt,Yt,options); 
          
    fprintf('%d: %s -> %s: Acc = %.4f\n',i,src,tar,acc);
    accuracy_list = [accuracy_list;acc];
end
