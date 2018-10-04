function [dwnOptimModel, dwnOptimCost] = transformDwnOptimisationModel(...
    dwnSystem, dwnObjective, treeData, dwnSysOption)
% This function transform the drinking water network(DWN) model provided in EFFINET into 
% optimisation form of (f + g)
%
% Syntax 
%   [dwnOptimisationModel, dwnOptimisationCost] = transformDwnForOptimisationModel(...
%    dwnSystem, dwnObjective, treeData, dwnSysOption)
%   
% INPUT 
%   dwnSystem     : EFFINET DWN system description
%   dwnObjective  : EFFINET DWN cost description
%   treeData      : scenario tree discription
%
% OUTPUT
%   dwnOptimisationModel : optimisation model form of DWN 
%   dwnOptimisationCost  : cost function model form of DWN 

dwnOptimModel.nu = dwnSystem.nu;
dwnOptimModel.nx = dwnSystem.nx;
dwnOptimModel.Np = dwnObjective.Hp;
nStage = size(treeData.stage, 1);

dwnOptimModel.xmin = dwnSystem.xmin;
dwnOptimModel.xmax = dwnSystem.xmax;
dwnOptimModel.xs = dwnObjective.xs;

dwnOptimModel.gamma_xbox = dwnSysOption.gamma_xbox;
dwnOptimModel.gamma_xs = dwnSysOption.gamma_xs;

% Null spaces and particular soultion format
dwnOptimModel.L = null(dwnSystem.E);
dwnOptimModel.L1 = -pinv(dwnSystem.E)*(dwnSystem.Ed);

dwnOptimModel.A = dwnSystem.A;
dwnOptimModel.B = dwnSystem.B/3600;
dwnOptimModel.Gd = dwnSystem.Gd/3600;
dwnOptimModel.umin = 3600*dwnSystem.umin;
dwnOptimModel.umax = 3600*dwnSystem.umax;

% cost function
dwnOptimCost.Wu = dwnObjective.Wu;

% Normalise the constraints
if(dwnSysOption.cell)
    dwnOptimModel.F = cell(nStage,1);
    dwnOptimModel.G = cell(nStage,1);
    for j=1:nStage
        dwnOptimModel.F{j} = [eye(dwnSystem.nx);eye(dwnSystem.nx)];
        dwnOptimModel.G{j} = eye(dwnSystem.nu);
    end
else
    dwnOptimModel.F = [eye(dwnSystem.nx);eye(dwnSystem.nx)];
    dwnOptimModel.G = eye(dwnSystem.nu);
end
dwnOptimModel.cell = dwnSysOption.cell;
dwnOptimModel.normalized = dwnSysOption.normalise;

dwnOptimModel.umin = kron(ones(nStage,1),dwnOptimModel.umin);
dwnOptimModel.umax = kron(ones(nStage,1),dwnOptimModel.umax);

dwnOptimModel.xmin = kron(ones(nStage,1),dwnOptimModel.xmin);
dwnOptimModel.xmax = kron(ones(nStage,1),dwnOptimModel.xmax);
dwnOptimModel.xs = kron(ones(nStage,1),dwnOptimModel.xs);

end



