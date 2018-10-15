function [ dwnSmpcZvar, details] = calculateDwnSolveStepVersion(dwnOptimModel, treeData,...
    dualY, dwnFactorStepModel, xInitialState, optionSolveStep)
% This function calcualte the dual gradient for the smpc optimisaiton problem for the 
% DWN. This step uses the offline matrices from the factor step   
% 
% syntax 
%
% INPUT 
%   dwnOptimModel     :  smpc optimisation model of the dwn  
%   treeData    :  uncertainty tree structure 
%   dwnFactorStepModel   :  Cost function
%   dualY     :   dual variable 
%   xInitialState     :  x_0 = p; v_{-1} = q; \hat{v}_{-1} = \hat{q};
%
% OUTPUT 
%   dwnSmpcZvar  : X states 
%                : U control/Input
%  details  :
%

nNode = size(treeData.stage, 1);% total number of nodes including leaves 
nLeave = size(treeData.leaves, 1);% total number of leaves
nv = size(dwnOptimModel.L, 2);

dwnSmpcZvar.X = zeros(dwnOptimModel.nx, nNode + 1);
dwnSmpcZvar.U = zeros(dwnOptimModel.nu, nNode);
v = zeros(nv, nNode);

q = zeros(dwnOptimModel.nx, nNode + 1);
r = zeros(nv, nNode + 1);
sigma = zeros(nv, nNode + 1);

for k = dwnOptimModel.Np:-1:1
    nodesStage = find(treeData.stage == k-1);
    if(k == 1)
        sigma(:, 1) = r(:, 2) + optionSolveStep.beta(:, 1);
        v(:, 1) = dwnFactorStepModel.Phi{1}*dualY.x(:, 1) + dwnFactorStepModel.Psi{1}*...
            dualY.u(:, 1) + dwnFactorStepModel.Theta{1}*q(:, 2) +...
            dwnFactorStepModel.omega{1}*sigma(:, 1);
        
        r(:, 1) = dwnFactorStepModel.K{1}*sigma(:,1) + dwnFactorStepModel.d{1}*...
            dualY.x(:,1) + dwnFactorStepModel.f{1}*dualY.u(:,1) +...
            dwnFactorStepModel.g{1}*q(:,2);
        q(:, 1) = dwnOptimModel.A'*dwnOptimModel.F{1}'*dualY.x(:,1) + dwnOptimModel.A'*q(:,2);
    else
        Pnodes_stage = find(treeData.stage == k-2);
        for j = 1:length(nodesStage)
            sigma(:, nodesStage(j)) = r(:,nodesStage(j)+1) + optionSolveStep.beta(:,nodesStage(j));
            v(:,nodesStage(j)) = dwnFactorStepModel.Phi{nodesStage(j)}*dualY.x(:,nodesStage(j)) +...
                dwnFactorStepModel.Psi{nodesStage(j)}*dualY.u(:,nodesStage(j)) +...
                dwnFactorStepModel.Theta{nodesStage(j)}*q(:,nodesStage(j)+1) +...
                dwnFactorStepModel.omega{nodesStage(j)}*sigma(:,nodesStage(j));
        end
        
        for j = 1:length(Pnodes_stage)
            nChild = treeData.children{Pnodes_stage(j)};
            for l = 1:length(nChild)
                if(l == 1)
                    r(:, Pnodes_stage(j) + 1) = dwnFactorStepModel.K{nChild(l)}*sigma(:,nChild(l)) +...
                        dwnFactorStepModel.d{nChild(l)}*dualY.x(:,nChild(l)) +...
                        dwnFactorStepModel.f{nChild(l)}*dualY.u(:,nChild(l)) +...
                        dwnFactorStepModel.g{nChild(l)}*q(:,nChild(l)+1);
                    q(:,Pnodes_stage(j) + 1) = dwnOptimModel.A'*dwnOptimModel.F{nChild(l)}'*...
                        dualY.x(:,nChild(l)) + dwnOptimModel.A'*q(:,nChild(l) + 1);
                else
                    r(:,Pnodes_stage(j) + 1) = dwnFactorStepModel.K{nChild(l)}*sigma(:,nChild(l)) +...
                        dwnFactorStepModel.d{nChild(l)}*dualY.x(:,nChild(l)) +...
                        dwnFactorStepModel.f{nChild(l)}*dualY.u(:,nChild(l)) +...
                        dwnFactorStepModel.g{nChild(l)}*q(:,nChild(l) + 1) + r(:,Pnodes_stage(j) + 1);
                    q(:,Pnodes_stage(j) + 1) = dwnOptimModel.A'*dwnOptimModel.F{nChild(l)}'*...
                        dualY.x(:,nChild(l)) + dwnOptimModel.A'*q(:,nChild(l) + 1) + q(:,Pnodes_stage(j) + 1);
                end
            end
        end
    end 
end

details.v = v;
dwnSmpcZvar.X(:, 1) = xInitialState;

for kk = 1:nNode-nLeave + 1
    if(kk == 1)
        v(:, kk) = dwnFactorStepModel.K{kk, 1}*optionSolveStep.v + v(:, kk);
        dwnSmpcZvar.U(:, kk) = dwnOptimModel.L*v(:, kk) + optionSolveStep.vhat(:, kk);
        dwnSmpcZvar.X(:,kk + 1) = dwnOptimModel.A*dwnSmpcZvar.X(:, 1) + dwnOptimModel.B*dwnSmpcZvar.U(:, kk) + ...
            optionSolveStep.w(:, kk);
    else
        nChild = treeData.children{kk-1};
        for l = 1:length(nChild)
            v(:, nChild(l)) = dwnFactorStepModel.K{nChild(l), 1}*v(:, treeData.ancestor(nChild(l), 1)) +...
                v(:, nChild(l));
            dwnSmpcZvar.U(:, nChild(l)) = dwnOptimModel.L*v(:, nChild(l)) + optionSolveStep.vhat(:, nChild(l));
            dwnSmpcZvar.X(:, nChild(l) + 1) = dwnOptimModel.A*dwnSmpcZvar.X(:,kk) + dwnOptimModel.B*dwnSmpcZvar.U(:,nChild(l)) +...
                optionSolveStep.w(:, nChild(l));
        end
    end
end

details.v1 = v;
details.q = q;
end





