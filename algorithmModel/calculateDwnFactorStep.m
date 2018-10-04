function [ dwnFactorStepModel ] = calculateDwnFactorStep( dwnOptimModel, dwnOptimCost, treeData)
% 
% This function calculate the offline matrices used for calculating the 
% dual gradient of the MPC optimisation problem for DWN  
%
% Syntax 
%     [ dwnFactorStepModel ] = calculateDwnFactorStep( dwnOptimModel, dwnOptimCost, treeData)
% 
% INPUT 
%   dwnOptimModel  :
%   dwnOptimCost   :
%   treeData       :
%
% OUTPUT 
%  dwnFactorStepModel  :
%

dwnFactorStepModel = struct('P', cell(1,1), 'K', cell(1,1), 'Phi', cell(1,1),...
    'omega', cell(1,1), 'd', cell(1,1), 'f', cell(1,1));

Wv = dwnOptimModel.L'*dwnOptimCost.Wu*dwnOptimModel.L;
nv = size(dwnOptimModel.L,2); %reduced variable

if(dwnOptimModel.cell)
    Bbar = dwnOptimModel.B*dwnOptimModel.L;
    
    for k = dwnOptimModel.Np:-1:1
        nodesStage = find(treeData.stage==k-1);
        for j = 1:length(nodesStage)
            Wv_k = treeData.prob(nodesStage(j))*Wv;
            
            dwnFactorStepModel.Gbar{nodesStage(j), 1} = dwnOptimModel.G{nodesStage(j),...
                1}*dwnOptimModel.L;
            
            dwnFactorStepModel.omega{nodesStage(j), 1} = -0.5*(Wv_k\eye(nv));
            dwnFactorStepModel.Phi{nodesStage(j), 1} = dwnFactorStepModel.omega{nodesStage(j),...
                1}*(dwnOptimModel.F{nodesStage(j)}*Bbar)';
            dwnFactorStepModel.Psi{nodesStage(j), 1} = dwnFactorStepModel.omega{nodesStage(j),...
                1}*dwnFactorStepModel.Gbar{nodesStage(j)}';
            dwnFactorStepModel.Theta{nodesStage(j), 1} = dwnFactorStepModel.omega{nodesStage(j),...
                1}*Bbar';
            
            dwnFactorStepModel.K{nodesStage(j), 1} = eye(nv);
            dwnFactorStepModel.d{nodesStage(j), 1} = dwnFactorStepModel.K{nodesStage(j),...
                1}'*(dwnOptimModel.F{nodesStage(j)}*Bbar)';
            dwnFactorStepModel.f{nodesStage(j), 1} = dwnFactorStepModel.K{nodesStage(j),...
                1}'*dwnFactorStepModel.Gbar{nodesStage(j)}';
            dwnFactorStepModel.g{nodesStage(j), 1} = dwnFactorStepModel.K{nodesStage(j),...
                1}'*Bbar';
        end
        
        if(k==1)
            dwnFactorStepModel.P{1} = -Wv;
        else
            factorStepNodesStage = find(treeData.stage == k-2);
            for j = 1:length(factorStepNodesStage)
                nChild = treeData.children{factorStepNodesStage(j)};
                dwnFactorStepModel.P{factorStepNodesStage(j)+1, 1} = -sum(treeData.prob(nChild))*Wv;
            end
        end
    end
    dwnFactorStepModel.Bbar = dwnOptimModel.B*dwnOptimModel.L;
else
    Gbar = dwnOptimModel.G*dwnOptimModel.L;
    Bbar = dwnOptimModel.B*dwnOptimModel.L;
    Fbar = dwnOptimModel.F*Bbar;
    
    for k = dwnOptimModel.Np:-1:1
        nodesStage = find(treeData.stage == k-1);
        for j = 1:length(nodesStage)
            if(k == dwnOptimModel.Np)
                Wv_k = treeData.prob(nodesStage(j))*Wv;
            else
                Wv_k = treeData.prob(nodesStage(j))*Wv + dwnFactorStepModel.P{nodesStage(j) + 1};
                nChild = treeData.children{nodesStage};
                for l = 1:length(nChild)
                    Wv_k = Wv_k+treeData.prob(nChild(l))*Wv_k;
                end
            end
            dwnFactorStepModel.omega{nodesStage(j), 1} = -0.5*(Wv_k\eye(nv));
            dwnFactorStepModel.Phi{nodesStage(j), 1} = dwnFactorStepModel.omega{nodesStage(j),...
                1}*(Fbar + Gbar)';
            dwnFactorStepModel.Theta{nodesStage(j), 1} = dwnFactorStepModel.omega{nodesStage(j),...
                1}*Bbar';
            dwnFactorStepModel.K{nodesStage(j), 1} = -2*treeData.prob(nodesStage(j))*...
                (dwnFactorStepModel.omega{nodesStage(j), 1}*Wv);
            dwnFactorStepModel.d{nodesStage(j), 1} = dwnFactorStepModel.K{nodesStage(j), 1}'*...
                (Fbar + Gbar)';
            dwnFactorStepModel.f{nodesStage(j), 1} = dwnFactorStepModel.K{nodesStage(j), 1}'*Bbar';
        end
        if(k == 1)
            dwnFactorStepModel.P{1} = -Wv*dwnFactorStepModel.K{1};
        else
            factorStepNodesStage = find(treeData.stage == k-2);
            for j = 1:length(factorStepNodesStage)
                nChild = treeData.children{factorStepNodesStage(j)};
                for l = 1:length(nChild)
                    if(l == 1)
                        dwnFactorStepModel.P{factorStepNodesStage(j) + 1, 1} = treeData.prob(nChild(l))*...
                            dwnFactorStepModel.K{nChild(l), 1};
                    else
                        dwnFactorStepModel.P{factorStepNodesStage(j) + 1, 1} = dwnFactorStepModel.P{...
                            factorStepNodesStage(j)+1, 1} + treeData.prob(nChild(l))*dwnFactorStepModel.K{...
                            nChild(l),1};
                    end
                end
                dwnFactorStepModel.P{factorStepNodesStage(j) + 1, 1} = -Wv*dwnFactorStepModel.P{...
                    factorStepNodesStage(j) + 1, 1};
            end
        end
    end
    dwnFactorStepModel.Bbar = Bbar;
    dwnFactorStepModel.Gbar = Gbar;
end

end



