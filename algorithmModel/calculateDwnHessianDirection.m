function [ hessianDir, details] = calculateDwnHessianDirection(dwnOptimModel, treeData,...
    dualDir, dwnFactorStepModel)
% This function is the oracle that calculate the hessian-direction product for the dual fbe-envelop 
% for the smpc optimisaiton problem for the DWN. This step uses the offline matrices from the factor step   
% 
% syntax 
%    [ hessianDir, details] = calculateDwnHessianDirection(dwnOptimModel, treeData,...
%          dwnFactorStepModel, dualDir)
%
% INPUT 
%   dwnOptimModel     :  smpc optimisation model of the dwn  
%   treeData    :  uncertainty tree structure 
%   dwnFactorStepModel   :  Cost function
%   dualDir     :   direction for the direction update 
%   xInitialState     :  x_0 = p; v_{-1} = q; \hat{v}_{-1} = \hat{q};
%
% OUTPUT 
%   hessianDir  : X states 
%               : U control/Input
%  details   :
%   calculateDwnSolveStep(dwnOptimModel, treeData, dualY, dwnFactorStepModel, xInitialState, optionSolveStep)

nNode = size(treeData.stage, 1);% total number of nodes including leaves 
nLeave = size(treeData.leaves, 1);% total number of leaves
nv = size(dwnOptimModel.L, 2);

hessianDir.X = zeros(dwnOptimModel.nx, nNode + 1);
hessianDir.U = zeros(dwnOptimModel.nu, nNode);
v = zeros(nv, nNode);

q = zeros(dwnOptimModel.nx, nNode + 1);
r = zeros(nv, nNode + 1);
sigma = zeros(nv, nNode + 1);

for k = dwnOptimModel.Np:-1:1
    nodesStage = find(treeData.stage == k-1);
    if(k == 1)
        v(:, 1) = dwnFactorStepModel.Phi{1}*dualDir.x(:, 1) + dwnFactorStepModel.Psi{1}*...
            dualDir.u(:, 1) + dwnFactorStepModel.Theta{1}*q(:, 2) +...
            dwnFactorStepModel.omega{1}*r(:, 2);
        
        r(:, 1) = dwnFactorStepModel.K{1}*r(:, 2) + dwnFactorStepModel.d{1}*...
            dualDir.x(:,1) + dwnFactorStepModel.f{1}*dualDir.u(:,1) +...
            dwnFactorStepModel.g{1}*q(:,2);
        q(:, 1) = dwnOptimModel.A'*dwnOptimModel.F{1}'*dualDir.x(:,1) + dwnOptimModel.A'*q(:,2);
    else
        for j = 1:length(nodesStage)
            sigma(:, nodesStage(j)) = r(:,nodesStage(j) + 1);
            v(:,nodesStage(j)) = dwnFactorStepModel.Phi{nodesStage(j)}*dualDir.x(:,nodesStage(j)) +...
                dwnFactorStepModel.Psi{nodesStage(j)}*dualDir.u(:, nodesStage(j)) +...
                dwnFactorStepModel.Theta{nodesStage(j)}*q(:, nodesStage(j)+1) +...
                dwnFactorStepModel.omega{nodesStage(j)}*r(:,nodesStage(j) + 1);
        end
        
        nodesPreviousStage = find(treeData.stage == k-2);
        for j = 1:length(nodesPreviousStage)
            nChild = treeData.children{nodesPreviousStage(j)};
            for l = 1:length(nChild)
                if(l == 1)
                    r(:, nodesPreviousStage(j) + 1) = dwnFactorStepModel.K{nChild(l)}*r(:,nChild(l) + 1) +...
                        dwnFactorStepModel.d{nChild(l)}*dualDir.x(:,nChild(l)) +...
                        dwnFactorStepModel.f{nChild(l)}*dualDir.u(:,nChild(l)) +...
                        dwnFactorStepModel.g{nChild(l)}*q(:,nChild(l)+1);
                    q(:,nodesPreviousStage(j) + 1) = dwnOptimModel.A'*dwnOptimModel.F{nChild(l)}'*...
                        dualDir.x(:,nChild(l)) + dwnOptimModel.A'*q(:,nChild(l) + 1);
                else
                    r(:,nodesPreviousStage(j) + 1) = dwnFactorStepModel.K{nChild(l)}*r(:,nChild(l) + 1) +...
                        dwnFactorStepModel.d{nChild(l)}*dualDir.x(:,nChild(l)) +...
                        dwnFactorStepModel.f{nChild(l)}*dualDir.u(:,nChild(l)) +...
                        dwnFactorStepModel.g{nChild(l)}*q(:,nChild(l) + 1) + r(:,nodesPreviousStage(j) + 1);
                    q(:,nodesPreviousStage(j) + 1) = dwnOptimModel.A'*dwnOptimModel.F{nChild(l)}'*...
                        dualDir.x(:,nChild(l)) + dwnOptimModel.A'*q(:,nChild(l) + 1) + q(:,nodesPreviousStage(j) + 1);
                end
            end
        end
    end 
end

details.v = v;
hessianDir.X(:, 1) = 0;

for iNode = 1:nNode - nLeave + 1
    if(iNode == 1)
        hessianDir.U(:, iNode) = dwnOptimModel.L*v(:, iNode);
        hessianDir.X(:,iNode + 1) = dwnOptimModel.B*hessianDir.U(:, iNode);
    else
        nChild = treeData.children{iNode-1};
        for l = 1:length(nChild)
            %if( nChild(l) == 8)
             %   nChild(l)
            %end
            v(:, nChild(l)) = dwnFactorStepModel.K{nChild(l)}*v(:, treeData.ancestor(nChild(l))) +...
                v(:, nChild(l));
            hessianDir.U(:, nChild(l)) = dwnOptimModel.L*v(:, nChild(l));
            hessianDir.X(:, nChild(l) + 1) = dwnOptimModel.A*hessianDir.X(:,iNode) + dwnOptimModel.B*hessianDir.U(:,nChild(l));
        end
    end
end

details.v1 = v;
details.q = q;
end





