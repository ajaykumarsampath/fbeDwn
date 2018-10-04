function [eliminateCurrentState] = calculateParticularSolution(dwnOptimModel, treeData,...
    dwnOptimCost, optionCurrentState)
%
% This function calculates the particular soultion for the equation Eu + E_d d = 0 
% given by u=-pinv(E)Ed d
% 
% INPUT 
%   dwnOptimModel        :  system dynamics
%   treeData             :  Tree structure
%   dwnOptimCost         :  Current demand estimated
%   optionparticularSol  :  Previous_vhat;
%
% Output 
%   cur_opt  :
%
% current_opts=par_sol_opt;

if(isfield(treeData, 'errorPrice'))
    ALPHA_FLAG = 1;
else
    ALPHA_FLAG = 0;
end

nStage = size(treeData.stage, 1);
nu = size(dwnOptimModel.L1, 1);
nv = size(dwnOptimModel.L, 2);

vhat = zeros(nu, nStage);
zeta = zeros(nu, nStage);
beta = zeros(nv, nStage);
w = zeros(dwnOptimModel.nx, nStage);

currentDemand = optionCurrentState.demand;
prevVhat = optionCurrentState.prev_vhat;

if(ALPHA_FLAG)
    alphaBar = zeros(nv, nStage);
else
    alphaBar = zeros(nv, dwnOptimModel.Np);
end

Wv1 = dwnOptimCost.Wu*dwnOptimModel.L;

for k = 1:dwnOptimModel.Np
    nodesStage = find(treeData.stage==k-1);
    if(~ALPHA_FLAG)
        alphaBar(:,k) = (dwnOptimCost.alpha(k,:)*dwnOptimModel.L)';
    end
    for j = 1:length(nodesStage)
        if(ALPHA_FLAG)
            ll = dwnOptimCost.alpha(k,:) + treeData.errorPrice(nodesStage(j), :);
            alphaBar(:, nodesStage(j)) = (ll*dwnOptimModel.L)';
        end
        w(:, nodesStage(j)) = dwnOptimModel.Gd*(treeData.value(nodesStage(j),:)' + currentDemand(k,:)');
        vhat(:, nodesStage(j)) = dwnOptimModel.L1*(currentDemand(k,:)' +...
            treeData.value(nodesStage(j),:)');
    end
end

deltaUhat = zeros(nu, nStage);
beta1 = zeros(nv, nStage);
for k = 1:dwnOptimModel.Np
    nodesStage = find(treeData.stage == k-1);
    if(k == 1)
        zeta(:,nodesStage) = (vhat(:,nodesStage)-prevVhat);
        deltaUhat(:, nodesStage) = zeta(:,nodesStage);
        nchild=treeData.children{nodesStage};
        for l=1:length(nchild)
            zeta(:,nodesStage) = zeta(:,nodesStage)-treeData.prob(nchild(l))*...
                (vhat(:,nchild(l))-vhat(:,nodesStage));
        end
        beta(:,nodesStage) = alphaBar(:,1)+2*(zeta(:,nodesStage)'*Wv1)';
        beta1(:, nodesStage) = 2*(zeta(:,nodesStage)'*Wv1)';
    else
        for j = 1:length(nodesStage)
            zeta(:,nodesStage(j)) = treeData.prob(nodesStage(j))*(vhat(:,nodesStage(j)) -...
                vhat(:,treeData.ancestor(nodesStage(j))));
            deltaUhat(:, nodesStage(j)) = vhat(:,nodesStage(j)) -...
                vhat(:,treeData.ancestor(nodesStage(j)));
            if(k < dwnOptimModel.Np)
                nchild = treeData.children{nodesStage(j)};
                for l=1:length(nchild)
                    zeta(:,nodesStage(j)) = zeta(:,nodesStage(j)) -...
                        treeData.prob(nchild(l))*(vhat(:,nchild(l)) - vhat(:,nodesStage(j)));
                end
            end
            if(ALPHA_FLAG)
                beta(:,nodesStage(j)) = treeData.prob(nodesStage(j))*alphaBar(:,nodesStage(j)) +...
                    2*(zeta(:,nodesStage(j))'*Wv1)';
                beta1(:, nodesStage(j)) = 2*(zeta(:,nodesStage(j))'*Wv1)';
            else
                beta(:,nodesStage(j)) = treeData.prob(nodesStage(j))*alphaBar(:,k) +...
                    2*(zeta(:,nodesStage(j))'*Wv1)';
            end
        end
    end
end

eliminateCurrentState = optionCurrentState;
eliminateCurrentState.beta = beta;
eliminateCurrentState.vhat = vhat;
eliminateCurrentState.alpha_bar = alphaBar;
eliminateCurrentState.w = w;
eliminateCurrentState.deltaUhat = deltaUhat; 
eliminateCurrentState.beta1 = beta1;
eliminateCurrentState.zeta = zeta;
eliminateCurrentState.Wv1 = Wv1;
end
