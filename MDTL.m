function [Acc] = MDTL(Xs,Ys,Xt,Yt,options)
%% Inputs:
%%% Xs      : Source domain feature matrix, ns * dim
%%% Ys      : Source domain label matrix, ns * 1
%%% Xt      : Target domain feature matrix, nt * dim
%%% Yt      : Target domain label matrix, nt * 1 (only used for testing accuracy)
%%% options : algorithm options:
%%%%% options.d      : #subspace dimension, default: 20   
%%%%% options.T      : #iterations, default: 10
%%%%% options.p      : #neighbors in manifold regularization, default: 5
%%%%% options.gamma  : gamma in paper, default: 0.01
%%%%% options.eta    : eta in paper, default: 10
%%%%% options.rho    : rho in paper, default: 1
%%%%% options.mu     : mu in paper, default: 0.1
%% Outputs:
%%%% Acc      :  Final accuracy value


%% Algorithm starts here
    fprintf('MDTL starts...\n');
    
    %% Load algorithm options
    if ~isfield(options,'d')
        options.d = 20;
    end
    if ~isfield(options,'T')
        options.T = 10;
    end
    if ~isfield(options,'p')
        options.p = 5;
    end
    if ~isfield(options,'gamma')
        options.gamma = 0.01;
    end
    if ~isfield(options,'eta')
        options.eta = 10;
    end
    if ~isfield(options,'rho')
        options.rho = 1;
    end
    if ~isfield(options,'mu')
        options.mu = 0.1;
    end

    %% Manifold subspace learning
    [Xs, Xt] = GFK_Map(Xs,Xt,options.d);  % n*m
    Xs = double(Xs)';  % m*n
    Xt = double(Xt)';

    X = [Xs,Xt];  % m*n
    ns = size(Xs,2);
    nt = size(Xt,2);
    n = ns + nt; 
    C = length(unique(Ys));
    acc_iter = [];
   
    YY = [];
    for c = 1 : C
        YY = [YY,Ys==c];
    end
    YY = [YY;zeros(nt,C)];

    %% Data normalization
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));  % m*n

    %% Manifold regulization 
    if options.rho > 0
        manifold.k = options.p;
        manifold.Metric = 'Cosine';
        manifold.NeighborMode = 'KNN';
        manifold.WeightMode = 'Cosine';
        W = lapgraph(X',manifold);
        Dw = diag(sparse(sqrt(1 ./ sum(W))));
        L = eye(n) - Dw * W * Dw;
    else
        L = 0;
    end
    
    % Generate presudo labels for the target domain
    knn_model = fitcknn(X(:,1:ns)',Ys,'NumNeighbors',1);
    Yt0 = knn_model.predict(X(:,ns + 1:end)');

    % Construct kernel
    K = Kernel('rbf',X,sqrt(sum(sum(X .^ 2).^0.5)/n));
    A = diag(sparse([ones(ns,1);zeros(nt,1)]));  %  diagonal domain indicator matrix
       
    for t = 1 : options.T       
       %%  CLDA MMD
        e = [1/ns*ones(ns,1); -1/nt*ones(nt,1)];
        [Ys_onehot] = one_hot_encoding(Ys,C);
        Ns = Ys_onehot/ns;
        Nt = zeros(nt,C);

        % construct MMD matrix
        if ~isempty(Yt0) && length(Yt0)==nt
            [Yt0_onehot] = one_hot_encoding(Yt0,C);
            Nt = Yt0_onehot/nt;
        end
        %%% Mt: d_intra, feature transferability
        Mt = [Ns*Ns', -Ns*Nt';  -Nt*Ns', Nt*Nt'];
        %%% Md: d_inter, feature discriminability
        Md = e*e'*C; 
        for i = 1:C
            Ws = repmat(Ns(:,i),1,C-1);
            Wt = [Nt(:,1:i-1),Nt(:,i+1:end)];
            Md = Md + [Ws*Ws', -Ws*Wt';  -Wt*Ws', Wt*Wt'];
        end
        Mt = Mt / norm(Mt,'fro');
        Md = Md / norm(Md,'fro');
        mu = options.mu;
        M = Mt-mu*Md;
        
        %% Compute coefficients vector Alpha
        Alpha = ((A + options.eta * M + options.rho * L) * K + options.gamma * speye(n, n)) \ (A * YY);  % SRM+CLDA+MR
        
        F = K * Alpha;
        [~,Yt0] = max(F,[],2);

        %% Compute accuracy
        Acc = numel(find(Yt0(ns+1:end)==Yt)) / nt;
        Yt0 = Yt0(ns+1:end);
        acc_iter = [acc_iter;Acc];
        fprintf('Iteration:[%d/10], Acc=%f\n',t,Acc);
    end
    fprintf('MDTL ends!\n');
end


function K = Kernel(ker,X,sigma)  
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
end


function [Y_onehot] = one_hot_encoding(Y, C)
    % intput:
    % Y: n*1, label vector
    % C: number of source class, C=length(unique(Ys))
    % output£º
    % Y_onehot£º n*C, one-hot label matrix

    n = length(Y); % number of samples
    Y_onehot = zeros(n, C);
    for i = 1:n
        class = Y(i);
        Y_onehot(i, class) = 1;
    end
end

