function [  dwnSmpcZvar, detailsAlgoApg] = solveSmpcDwnWithApg(dwnOptimModel, dwnFactorStepModel,...
    treeData, dwnOptimCost, optionDwnApg) 
% APG_effinet_dist_box 
% This function is implements the dual proximal gradient algorithm to solve 
% the sc-mpc of the DWN with soft constraints on state and hard constraits
% on input.
%
% Syntax 
%  
% INPUT   
%   dwnOptimalModel      :  dwn model for the smpc optimisation 
%   dwnFactorStepModel   :  offline matrices for factor step
%   treeData             :  uncertainty data 
%   dwnOptimCost         :  cost function
%   optionDwnApg         :  options for the dual APG algorithm
%             Termination options: maximum number of steps etc   
%             Type of constraints: soft or hard constaints
%             Intial condition and preious control input.
% 


nNodes = length(treeData.stage);%toal nodes in the tree
nDualx = size(dwnOptimModel.F{1}, 1);
nDualu = size(dwnOptimModel.G{1}, 1);

optionProximal.lambda = optionDwnApg.lambda;
optionProximal.xmin = reshape(dwnOptimModel.xmin, dwnOptimModel.nx, nNodes);
optionProximal.xs = reshape(dwnOptimModel.xs, dwnOptimModel.nx, nNodes);
optionProximal.xmax = reshape(dwnOptimModel.xmax, dwnOptimModel.nx, nNodes);
optionProximal.umin = reshape(dwnOptimModel.umin, dwnOptimModel.nu, nNodes);
optionProximal.umax = reshape(dwnOptimModel.umax, dwnOptimModel.nu, nNodes);
optionProximal.nx = dwnOptimModel.nx;
optionProximal.nu = dwnOptimModel.nu;
optionProximal.gamma_xbox = dwnOptimModel.gamma_xbox/optionDwnApg.lambda;
optionProximal.gamma_xs = dwnOptimModel.gamma_xs/optionDwnApg.lambda;
optionProximal.iter = 200;
optionProximal.constraints = optionDwnApg.constraints;

detailsAlgoApg.terminationCriteria = zeros(1,4);

dualVarPrevY.x = zeros(nDualx, nNodes);
dualVarPrevY.u = zeros(nDualu, nNodes);

dualVarY.x = zeros(nDualx, nNodes);
dualVarY.u = zeros(nDualu, nNodes);

dualVarW.x = zeros(nDualx, nNodes);
dualVarW.u = zeros(nDualu, nNodes);

primalFeasible.x = zeros(nDualx, nNodes);
primalFeasible.u = zeros(nDualu, nNodes);

uPrev = dwnOptimModel.L*optionDwnApg.state.v + optionDwnApg.state.prev_vhat;
theta = [1 1]'; 

tic
iStep = 1;
while( iStep < optionDwnApg.steps )
    % step 1: accelerating dual vector
    dualVarW.x = dualVarY.x + theta(2)*(1/theta(1) - 1)*(dualVarY.x - dualVarPrevY.x);
    dualVarW.u = dualVarY.u + theta(2)*(1/theta(1) - 1)*(dualVarY.u - dualVarPrevY.u);
    % step 2: calculate the gradient of the conjugate f
    dwnSmpcZvar = calculateDwnSolveStep(dwnOptimModel, treeData, dualVarW, dwnFactorStepModel,...
        optionDwnApg.x, optionDwnApg.state);
    % step 3 : calculate the proximal of g
    [dwnSmpcTvar, proximalCost] = calculateDwnProximalStep(dwnOptimModel, treeData, dualVarW,...
        dwnSmpcZvar, optionProximal);
    % step 4 : calculate the residual of global Fbe and gradient of the dual Fbe
    dwnSmpcResidual = calculateDwnFbeResidual(dwnOptimModel, dwnSmpcZvar, dwnSmpcTvar);
    
    optionDwnApgObjective.previousU = uPrev;
    optionDwnApgObjective.lambda = optionDwnApg.lambda;
    optionDwnApgObjective.dualVar = dualVarW;
    optionDwnApgObjective.dwnSmpcZvar = dwnSmpcZvar;
    optionDwnApgObjective.proximalCost = proximalCost;
    optionDwnApgObjective.fbeResidual = dwnSmpcResidual;
    residual = [dwnSmpcResidual.x; dwnSmpcResidual.u];
    dimResidual = size(residual);
    
    fbeObjDualY = calculateDwnDualEnvelopObjective( dwnOptimCost, treeData, optionDwnApgObjective);
    detailsAlgoApg.fbeObjDualY(iStep) = fbeObjDualY;
    detailsAlgoApg.normSmpcResidual(iStep) = norm(reshape(residual, dimResidual(1)*dimResidual(2), 1));
    
    dualVarPrevY = dualVarY;
    dualVarY.x = dualVarW.x - optionDwnApg.lambda*dwnSmpcResidual.x;
    dualVarY.u = dualVarW.u - optionDwnApg.lambda*dwnSmpcResidual.u;
    
    iStep = iStep + 1;
    theta(1) = theta(2);
    theta(2) = (sqrt(theta(1)^4 + 4*theta(1)^2) - theta(1)^2)/2; 
    
end

detailsAlgoApg.gpad_solve = toc;
detailsAlgoApg.dualVarW = dualVarW;
detailsAlgoApg.dualVar = dualVarY;
detailsAlgoApg.dwnSmpcTvar = dwnSmpcTvar;

end







